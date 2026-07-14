"""Minimal stdlib PNG writer for frame evidence (render -> artifact)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

from .vic import PALETTE


def write_png_rgb(path: str | Path, width: int, height: int, rgb: bytes) -> None:
    if len(rgb) != width * height * 3:
        raise ValueError("rgb buffer size mismatch")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter: none
        raw += rgb[y * stride:(y + 1) * stride]
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(png)


def indexed_to_rgb(indexed: bytes | bytearray, *, scale: int = 1,
                   width: int | None = None) -> bytes:
    """Expand C64 color indices to RGB, optionally integer-upscaled.
    ``width`` is required when scale > 1."""
    if scale == 1:
        out = bytearray(len(indexed) * 3)
        for i, ci in enumerate(indexed):
            out[i * 3: i * 3 + 3] = bytes(PALETTE[ci & 0x0F])
        return bytes(out)
    if width is None:
        raise ValueError("width required when scaling")
    height = len(indexed) // width
    out = bytearray(width * scale * height * scale * 3)
    ow = width * scale
    for y in range(height):
        rowbase = y * width
        for x in range(width):
            r, g, b = PALETTE[indexed[rowbase + x] & 0x0F]
            for sy in range(scale):
                o = ((y * scale + sy) * ow + x * scale) * 3
                for sx in range(scale):
                    out[o + sx * 3] = r
                    out[o + sx * 3 + 1] = g
                    out[o + sx * 3 + 2] = b
    return bytes(out)


def save_frame_png(path: str | Path, frame: tuple[int, int, bytes | bytearray],
                   *, scale: int = 2) -> None:
    """Save a VIC.render_frame() result as a PNG."""
    width, height, indexed = frame
    rgb = indexed_to_rgb(indexed, scale=scale, width=width)
    write_png_rgb(path, width * scale, height * scale, rgb)
