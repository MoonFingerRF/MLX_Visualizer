"""HTTP + WebSocket server hosting the viewer and streaming frames.

One asyncio server on one port: plain GETs serve the bundled viewer,
``GET /ws`` upgrades to a WebSocket. Runs entirely inside the
visualizer's worker event loop; ``broadcast`` is safe to call from any
coroutine on that loop, and slow clients are handled with per-client
send queues that drop stale snapshots instead of back-pressuring the
capture pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Set

from .websocket import OP_TEXT, accept_key, encode_frame, read_message

log = logging.getLogger("mlx_visualizer")

VIEWER_DIR = Path(__file__).parent / "viewer"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ico": "image/x-icon",
}

# Per-client outgoing queue depth. When a client can't keep up, the oldest
# queued snapshot for the same watch is superseded rather than piling up.
CLIENT_QUEUE_SIZE = 64


class Client:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.queue: "asyncio.Queue[tuple[int, bytes, int]]" = asyncio.Queue(CLIENT_QUEUE_SIZE)
        # Latest queued fingerprint per watch id, to coalesce stale frames.
        self.pending_watch: Dict[int, int] = {}
        self.alive = True

    def send(self, opcode: int, payload: bytes, watch_id: int = 0) -> None:
        """Queue a frame; drops the oldest frame when the queue is full."""
        while True:
            try:
                self.queue.put_nowait((opcode, payload, watch_id))
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass


class VizServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8791,
        on_message: Optional[Callable[["Client", dict], Awaitable[None]]] = None,
        hello: Optional[Callable[[], dict]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_message = on_message
        self.hello = hello
        self.clients: Set[Client] = set()
        self._server: Optional[asyncio.base_events.Server] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        addr = self._server.sockets[0].getsockname()
        self.port = addr[1]
        log.info("MLX Visualizer serving at http://%s:%s/", self.host, self.port)

    async def stop(self) -> None:
        for client in list(self.clients):
            client.alive = False
            client.writer.close()
        self.clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def has_clients(self) -> bool:
        return bool(self.clients)

    def broadcast_binary(self, payload: bytes, watch_id: int = 0) -> None:
        from .websocket import OP_BINARY

        for client in self.clients:
            client.send(OP_BINARY, payload, watch_id)

    def broadcast_json(self, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        for client in self.clients:
            client.send(OP_TEXT, data)

    # -- connection handling ------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                writer.close()
                return
            parts = request_line.decode("latin-1").split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = line.decode("latin-1").partition(":")
                headers[key.strip().lower()] = value.strip()

            if path.split("?")[0] == "/ws" and headers.get("upgrade", "").lower() == "websocket":
                await self._handle_ws(reader, writer, headers)
            else:
                await self._handle_http(writer, method, path)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception:
            log.exception("connection handler error")
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle_http(self, writer: asyncio.StreamWriter, method: str, path: str) -> None:
        path = path.split("?")[0]
        if path == "/":
            path = "/index.html"
        name = Path(path).name  # flat directory; defeats traversal
        target = VIEWER_DIR / name
        if method != "GET" or not target.is_file():
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found")
            await writer.drain()
            return
        body = target.read_bytes()
        ctype = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\nCache-Control: no-cache\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(head + body)
        await writer.drain()

    async def _handle_ws(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: dict) -> None:
        key = headers.get("sec-websocket-key")
        if not key:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n"
        ).encode("latin-1")
        writer.write(response)
        await writer.drain()

        client = Client(writer)
        # Queue hello before joining the broadcast set so it is always the
        # first message a client sees, even mid-capture-tick.
        if self.hello is not None:
            client.send(OP_TEXT, json.dumps(self.hello()).encode("utf-8"))
        self.clients.add(client)
        sender = asyncio.ensure_future(self._sender(client))
        try:
            while True:
                msg = await read_message(reader, writer)
                if msg is None:
                    break
                opcode, payload = msg
                if opcode == OP_TEXT and self.on_message is not None:
                    try:
                        obj = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        continue
                    await self.on_message(client, obj)
        except (ConnectionError, ValueError):
            pass
        finally:
            client.alive = False
            self.clients.discard(client)
            sender.cancel()

    async def _sender(self, client: Client) -> None:
        try:
            while client.alive:
                opcode, payload, _watch = await client.queue.get()
                client.writer.write(encode_frame(opcode, payload))
                await client.writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
