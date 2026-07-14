"""D64 disk image parsing + PRG helpers.

A D64 is a sector dump of a 1541 disk: 35 tracks (optionally 40), 256-byte
sectors, directory on track 18.  This module gives the framework's drive
HLE (KERNAL LOAD) and the porting agent's inspection tools a clean view:
directory entries, file extraction by pattern, and the BASIC-stub ``SYS``
parser used to find a PRG's machine-code entry point.

Error-image variants (with per-sector error bytes appended) are accepted;
the error data is ignored.  GCR-level images (G64) are not supported —
that is a different evidence tier, added only when a real protection needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SECTOR_SIZE = 256
# sectors per track, 1-indexed; tracks 36-40 for extended images
SECTORS_PER_TRACK = {
    t: (21 if t <= 17 else 19 if t <= 24 else 18 if t <= 30 else 17)
    for t in range(1, 41)
}
_SIZES = {
    683 * 256: 35, 683 * 256 + 683: 35,      # 35 tracks (+error bytes)
    768 * 256: 40, 768 * 256 + 768: 40,      # 40 tracks (+error bytes)
}

FILE_TYPES = {0: "DEL", 1: "SEQ", 2: "PRG", 3: "USR", 4: "REL"}


@dataclass(frozen=True)
class DirEntry:
    name: bytes          # PETSCII, $A0 padding stripped
    file_type: str
    closed: bool
    track: int
    sector: int
    blocks: int

    @property
    def display_name(self) -> str:
        return self.name.decode("latin-1")


class DiskImage:
    def __init__(self, data: bytes) -> None:
        if len(data) not in _SIZES:
            raise ValueError(
                f"unsupported D64 size {len(data)} "
                f"(expected one of {sorted(_SIZES)})"
            )
        self.tracks = _SIZES[len(data)]
        self.data = bytes(data)

    @classmethod
    def load(cls, path: str | Path) -> "DiskImage":
        return cls(Path(path).read_bytes())

    def _offset(self, track: int, sector: int) -> int:
        if not 1 <= track <= self.tracks:
            raise ValueError(f"track {track} out of range 1..{self.tracks}")
        if sector >= SECTORS_PER_TRACK[track]:
            raise ValueError(f"sector {sector} out of range on track {track}")
        off = 0
        for t in range(1, track):
            off += SECTORS_PER_TRACK[t]
        return (off + sector) * SECTOR_SIZE

    def sector(self, track: int, sector: int) -> bytes:
        off = self._offset(track, sector)
        return self.data[off:off + SECTOR_SIZE]

    @property
    def disk_name(self) -> bytes:
        bam = self.sector(18, 0)
        return bam[0x90:0xA0].rstrip(b"\xA0")

    def directory(self) -> list[DirEntry]:
        entries: list[DirEntry] = []
        track, sec = 18, 1
        seen: set[tuple[int, int]] = set()
        while track:
            if (track, sec) in seen:
                raise ValueError("directory chain loops")
            seen.add((track, sec))
            raw = self.sector(track, sec)
            for i in range(8):
                e = raw[2 + 32 * i: 2 + 32 * i + 30]
                ftype = e[0]
                if ftype == 0:
                    continue
                entries.append(DirEntry(
                    name=e[3:19].rstrip(b"\xA0"),
                    file_type=FILE_TYPES.get(ftype & 0x07, f"?{ftype & 0x07}"),
                    closed=bool(ftype & 0x80),
                    track=e[1],
                    sector=e[2],
                    blocks=e[28] | (e[29] << 8),
                ))
            track, sec = raw[0], raw[1]
        return entries

    def read_chain(self, track: int, sector: int) -> bytes:
        """Follow a track/sector chain and return the file payload."""
        out = bytearray()
        seen: set[tuple[int, int]] = set()
        while track:
            if (track, sector) in seen:
                raise ValueError("file chain loops")
            seen.add((track, sector))
            raw = self.sector(track, sector)
            if raw[0] == 0:  # last sector: raw[1] = index of last used byte
                out += raw[2:raw[1] + 1]
                break
            out += raw[2:SECTOR_SIZE]
            track, sector = raw[0], raw[1]
        return bytes(out)

    def find(self, pattern: bytes | str) -> DirEntry:
        """CBM-DOS filename match: ``*`` suffix wildcard, ``?`` single char.
        ``*`` alone means the first file — the ``LOAD"*",8,1`` convention."""
        if isinstance(pattern, str):
            pattern = pattern.encode("latin-1")
        entries = self.directory()
        candidates = [e for e in entries if e.file_type != "DEL"]
        if pattern == b"*":
            if not candidates:
                raise FileNotFoundError("disk has no loadable files")
            return candidates[0]
        for e in candidates:
            if _cbm_match(pattern, e.name):
                return e
        names = ", ".join(repr(e.display_name) for e in candidates)
        raise FileNotFoundError(f"no file matching {pattern!r} (disk has: {names})")

    def read_file(self, pattern: bytes | str) -> bytes:
        e = self.find(pattern)
        return self.read_chain(e.track, e.sector)


def _cbm_match(pattern: bytes, name: bytes) -> bool:
    pi = 0
    for ni, ch in enumerate(name):
        if pi >= len(pattern):
            return False
        p = pattern[pi]
        if p == 0x2A:  # '*'
            return True
        if p != 0x3F and p != ch:  # '?'
            return False
        pi += 1
    if pi < len(pattern):
        return pattern[pi] == 0x2A and pi == len(pattern) - 1
    return True


# ---- PRG helpers -----------------------------------------------------------------

def prg_load_address(prg: bytes) -> int:
    if len(prg) < 3:
        raise ValueError("PRG too short")
    return prg[0] | (prg[1] << 8)


def prg_payload(prg: bytes) -> bytes:
    return prg[2:]


def parse_basic_sys(prg: bytes) -> int | None:
    """Extract the SYS target from a standard BASIC loader stub.

    A machine-code PRG published for ``LOAD"..",8,1 : RUN`` starts with one
    or more BASIC lines whose first statement is ``SYS<digits>`` (token $9E).
    Returns the SYS address, or None if the program does not start with a
    recognizable stub (then it is a raw machine-code PRG and the adapter
    must know its entry).  This parses the stub statically — no BASIC ROM
    is involved.
    """
    load = prg_load_address(prg)
    body = prg_payload(prg)
    if load != 0x0801 or len(body) < 6:
        return None
    pos = 0
    for _ in range(16):  # scan a bounded number of BASIC lines
        if pos + 4 > len(body):
            return None
        link = body[pos] | (body[pos + 1] << 8)
        if link == 0:
            return None
        p = pos + 4  # skip link + line number
        # skip leading spaces and colons
        while p < len(body) and body[p] in (0x20, 0x3A):
            p += 1
        if p < len(body) and body[p] == 0x9E:  # SYS token
            p += 1
            while p < len(body) and body[p] == 0x20:
                p += 1
            digits = bytearray()
            while p < len(body) and 0x30 <= body[p] <= 0x39:
                digits.append(body[p])
                p += 1
            if digits:
                return int(digits.decode("ascii"))
            return None
        # move to next line: link is an absolute RAM address
        nxt = link - 0x0801
        if nxt <= pos or nxt > len(body):
            return None
        pos = nxt
    return None
