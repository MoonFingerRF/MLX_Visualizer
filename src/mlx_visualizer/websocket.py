"""Minimal, dependency-free RFC 6455 WebSocket server primitives.

Only what the visualizer needs: server-side handshake, receiving masked
client frames (text/binary/ping/close, with continuation support), and
sending unmasked text/binary frames. Built on asyncio streams.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from asyncio import IncompleteReadError, StreamReader, StreamWriter
from typing import Optional, Tuple

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA

MAX_MESSAGE = 32 * 1024 * 1024


def accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes) -> bytes:
    """Server-to-client frame: FIN set, unmasked."""
    head = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([n])
    elif n < 1 << 16:
        head += bytes([126]) + struct.pack("!H", n)
    else:
        head += bytes([127]) + struct.pack("!Q", n)
    return head + payload


async def read_frame(reader: StreamReader) -> Tuple[int, bool, bytes]:
    """Read one client frame. Returns (opcode, fin, unmasked payload)."""
    b1, b2 = await reader.readexactly(2)
    fin = bool(b1 & 0x80)
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        (length,) = struct.unpack("!H", await reader.readexactly(2))
    elif length == 127:
        (length,) = struct.unpack("!Q", await reader.readexactly(8))
    if length > MAX_MESSAGE:
        raise ValueError("frame too large")
    mask = await reader.readexactly(4) if masked else b""
    payload = await reader.readexactly(length)
    if masked and length:
        data = bytearray(payload)
        for i in range(length):
            data[i] ^= mask[i & 3]
        payload = bytes(data)
    return opcode, fin, payload


async def read_message(reader: StreamReader, writer: StreamWriter) -> Optional[Tuple[int, bytes]]:
    """Read one complete message, transparently handling ping/pong and
    continuation frames. Returns (opcode, payload) or None on close."""
    message = bytearray()
    message_op = None
    while True:
        try:
            opcode, fin, payload = await read_frame(reader)
        except (IncompleteReadError, ConnectionError):
            return None
        if opcode == OP_CLOSE:
            try:
                writer.write(encode_frame(OP_CLOSE, payload[:2]))
                await writer.drain()
            except ConnectionError:
                pass
            return None
        if opcode == OP_PING:
            writer.write(encode_frame(OP_PONG, payload))
            await writer.drain()
            continue
        if opcode == OP_PONG:
            continue
        if opcode in (OP_TEXT, OP_BINARY):
            message_op = opcode
            message = bytearray(payload)
        elif opcode == OP_CONT and message_op is not None:
            message += payload
        else:
            raise ValueError(f"unexpected opcode {opcode}")
        if len(message) > MAX_MESSAGE:
            raise ValueError("message too large")
        if fin:
            return message_op, bytes(message)
