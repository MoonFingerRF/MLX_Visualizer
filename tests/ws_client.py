"""Tiny blocking WebSocket client used only by the test suite."""

from __future__ import annotations

import base64
import os
import socket
import struct
from typing import Optional, Tuple


class WSClient:
    def __init__(self, host: str, port: int, path: str = "/ws", timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("latin-1"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("handshake failed")
            response += chunk
        status = response.split(b"\r\n", 1)[0]
        if b"101" not in status:
            raise ConnectionError(f"unexpected handshake response: {status!r}")
        # Bytes past the header belong to the first WebSocket frame.
        self._buffer = response.split(b"\r\n\r\n", 1)[1]

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        if self._buffer:
            take, self._buffer = self._buffer[:n], self._buffer[n:]
            buf = take
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    def send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        n = len(payload)
        head = bytes([0x81])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack("!H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack("!Q", n)
        self.sock.sendall(head + mask + masked)

    def recv_message(self) -> Optional[Tuple[int, bytes]]:
        """Returns (opcode, payload) for the next text/binary message."""
        while True:
            b1, b2 = self._read_exact(2)
            opcode = b1 & 0x0F
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._read_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._read_exact(8))
            payload = self._read_exact(length) if length else b""
            if opcode == 0x8:
                return None
            if opcode in (0x9, 0xA):
                continue
            return opcode, payload

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
