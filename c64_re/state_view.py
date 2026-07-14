"""Byte-backed *typed views* over C64 RAM — the state-mirror machinery.

This is the generic half of the state-mirror pattern (dos_re's
docs/state_mirrors.md): recovered logic operates on a *view* (``view.wind``,
``view.slots[i].x``) and never sees an address; the game adapter's layout
module is the ONLY place its memory addresses are written down, as
:class:`StructView` subclasses built from these descriptors.

A view holds a **backend** (the ports-and-adapters seam) and its field
descriptors address the backend in flat 16-bit C64 addresses ($0000-$FFFF):

* :class:`ByteBackend` — reads/writes straight through a flat byte image
  (``bytes``/``bytearray``, e.g. a captured snapshot's ``ram`` blob or a
  native game state's byte image) at ``base + offset``, wrapping at 64 KB
  exactly like the 6502's address bus.
* :class:`RamBackend` — same, over the LIVE machine's 64 KB ``mem.ram``
  bytearray (RAM-direct; no PLA banking — mirrors address RAM, which is what
  game state lives in).
* :class:`OverlayBackend` — read-through overlay: reads fall through to a base
  reader, writes ACCUMULATE a ``{address: value}`` contract WITHOUT mutating
  the base — for contract-returning islands (whole-routine transforms
  returning a write set, verified against a golden).
* :class:`WidthContractBackend` — write-only ``{address: (value, width)}``
  accumulator for projection passes that read only original memory and emit a
  fresh write set.

Mapping from the dos_re original: ``ByteBackend`` and the contract backends
carry over unchanged in name and shape; addressing is flat 16-bit ($XXXX)
instead of segment:offset, so dos_re's ``SegmentBackend`` (a second base
translation for the game's other segments) has NO C64 counterpart and was
dropped — there is only one address space.  ``RamBackend`` is new here: the
C64 equivalent of viewing live VM memory (dos_re did that with ``ByteBackend``
over ``mem.data`` at ``ds << 4``; on the C64 the base is just an address).

Because all backends share one interface, the SAME view (and the same
recovered logic) runs over any of them — live VM RAM, a byte image, or an
accumulating contract for a golden test.

**Width-alias convention** (the "union" answer): when the ASM reads the same
bytes at different widths, give each width its OWN named field — a different
width is a different *semantic* (a velocity word vs an anim-mirror byte).
Same storage, two meanings, two names; never a width argument at the call
site.  16-bit fields are little-endian, as the 6502 stores its pointers.

Usage in a game adapter::

    from c64_re.state_view import RamBackend, StructView, StructArray, U8, U16, S16, coerce_backend

    class PlayerView(StructView):
        x    = U16(0)
        y    = U16(2)
        xvel = S16(6)

    class GameView(StructView):        # whole-RAM view: offsets ARE addresses
        wind  = U16(0xC0F6)            # $C0F6, from the oracle
        slots = StructArray(0x4F0A, 0x12, 40, PlayerView)

        def __init__(self, source):
            super().__init__(coerce_backend(source), 0)

Origin: ported from dos_re's ``dos_re/state_view.py`` (itself promoted from
the Prehistorik 2 port's ``pre2/bridge/dgroup_view.py``, where this machinery
carried the whole state-view migration byte-exactly).
"""
from __future__ import annotations


# ---- backends -----------------------------------------------------------------------------------------------

class ByteBackend:
    """Reads/writes go straight to a flat byte image at ``base + offset``.

    ``source`` is a ``bytes``/``bytearray`` (or anything indexable holding a
    64 KB-or-larger byte image, e.g. a snapshot's captured ``ram``).  ``base``
    is the $XXXX address the view's offset 0 maps to — derived from the
    oracle, never hard-coded in recovered logic.  Addresses wrap at 64 KB
    exactly like the 6502's address bus.
    """

    __slots__ = ("data", "base")

    def __init__(self, source, base: int = 0):
        self.data = source
        self.base = base & 0xFFFF

    def rb(self, off: int) -> int:
        return self.data[(self.base + off) & 0xFFFF]

    def wb(self, off: int, v: int) -> None:
        self.data[(self.base + off) & 0xFFFF] = v & 0xFF

    def rw(self, off: int) -> int:
        return self.rb(off) | (self.rb(off + 1) << 8)

    def ww(self, off: int, v: int) -> None:
        self.wb(off, v)
        self.wb(off + 1, v >> 8)


class RamBackend:
    """RAM-direct: reads/writes the LIVE machine's 64 KB ``ram`` bytearray at
    ``base + offset`` (wrapping at 64 KB).  ``source`` is a
    :class:`c64_re.memory.Memory` (anything with a ``.ram`` bytearray) or the
    64 KB bytearray itself.  This is deliberately the un-banked RAM view —
    game state lives in RAM, including under ROM — so a mirror never trips
    PLA banking or I/O side effects."""

    __slots__ = ("ram", "base")

    def __init__(self, source, base: int = 0):
        self.ram = source.ram if hasattr(source, "ram") else source
        self.base = base & 0xFFFF

    def rb(self, off: int) -> int:
        return self.ram[(self.base + off) & 0xFFFF]

    def wb(self, off: int, v: int) -> None:
        self.ram[(self.base + off) & 0xFFFF] = v & 0xFF

    def rw(self, off: int) -> int:
        return self.rb(off) | (self.rb(off + 1) << 8)

    def ww(self, off: int, v: int) -> None:
        self.wb(off, v)
        self.wb(off + 1, v >> 8)


class OverlayBackend:
    """Read-through overlay: reads fall through to ``base_rb(addr)`` unless already written;
    writes accumulate the ``writes`` contract (``{address: byte}``) and never touch the base.
    A contract-returning island runs its whole-routine transform over one of these and returns
    ``overlay.writes`` as its write set — the pass stays a pure function of its inputs."""

    __slots__ = ("_base_rb", "writes")

    def __init__(self, base_rb):
        self._base_rb = base_rb          # base_rb(addr) -> the ORIGINAL byte at a $XXXX address
        self.writes: dict[int, int] = {}

    def rb(self, off: int) -> int:
        o = off & 0xFFFF
        return self.writes[o] if o in self.writes else self._base_rb(o)

    def wb(self, off: int, v: int) -> None:
        self.writes[off & 0xFFFF] = v & 0xFF

    def rw(self, off: int) -> int:
        return self.rb(off) | (self.rb((off + 1) & 0xFFFF) << 8)

    def ww(self, off: int, v: int) -> None:
        self.wb(off, v)
        self.wb((off + 1) & 0xFFFF, v >> 8)


class WidthContractBackend:
    """A write-only contract accumulator emitting ``{address: (value, width)}`` — the
    width-tracking contract convention some islands use (vs :class:`OverlayBackend`'s
    byte-level ``{address: value}``).  Reads delegate to the island's own ``rb``/``rw``
    closures and do NOT see the accumulated writes — for projection passes that read
    only original memory and emit a fresh write set."""

    __slots__ = ("_rb", "_rw", "writes")

    def __init__(self, base_rb, base_rw):
        self._rb = base_rb
        self._rw = base_rw
        self.writes: dict[int, tuple[int, int]] = {}

    def rb(self, off: int) -> int:
        return self._rb(off & 0xFFFF)

    def rw(self, off: int) -> int:
        return self._rw(off & 0xFFFF)

    def wb(self, off: int, v: int) -> None:
        self.writes[off & 0xFFFF] = (v & 0xFF, 1)

    def ww(self, off: int, v: int) -> None:
        self.writes[off & 0xFFFF] = (v & 0xFFFF, 2)


# ---- field descriptors (offset RELATIVE to the view's base) -------------------------------------------------

class U16:
    """A little-endian 16-bit field."""

    def __init__(self, off: int):
        self.off = off

    def __get__(self, o, owner=None):
        if o is None:
            return self
        return o._backend.rw(o._base + self.off)

    def __set__(self, o, v: int):
        o._backend.ww(o._base + self.off, v)


class U8:
    """An 8-bit field."""

    def __init__(self, off: int):
        self.off = off

    def __get__(self, o, owner=None):
        if o is None:
            return self
        return o._backend.rb(o._base + self.off)

    def __set__(self, o, v: int):
        o._backend.wb(o._base + self.off, v)


class S16:
    """A little-endian *signed* 16-bit field (returns -0x8000..0x7FFF)."""

    def __init__(self, off: int):
        self.off = off

    def __get__(self, o, owner=None):
        if o is None:
            return self
        v = o._backend.rw(o._base + self.off)
        return v - 0x10000 if v & 0x8000 else v

    def __set__(self, o, v: int):
        o._backend.ww(o._base + self.off, v)


class S8:
    """An 8-bit *signed* field (returns -0x80..0x7F)."""

    def __init__(self, off: int):
        self.off = off

    def __get__(self, o, owner=None):
        if o is None:
            return self
        v = o._backend.rb(o._base + self.off)
        return v - 0x100 if v & 0x80 else v

    def __set__(self, o, v: int):
        o._backend.wb(o._base + self.off, v)


class U16Array:
    """A contiguous array of little-endian 16-bit words; ``view.field[i]`` reads/writes element ``i``."""

    def __init__(self, off: int, length: int):
        self.off = off
        self.length = length

    def __get__(self, o, owner=None):
        if o is None:
            return self
        return _U16ArrayView(o._backend, o._base + self.off, self.length)


class _U16ArrayView:
    __slots__ = ("_backend", "_base", "length")

    def __init__(self, backend, base: int, length: int):
        self._backend = backend
        self._base = base
        self.length = length

    def __getitem__(self, i: int) -> int:
        return self._backend.rw(self._base + i * 2)

    def __setitem__(self, i: int, v: int) -> None:
        self._backend.ww(self._base + i * 2, v)

    def __len__(self) -> int:
        return self.length


class StructArray:
    """A descriptor for a fixed-stride array of structs; ``view.field[i]`` returns ``struct_cls`` bound to
    ``base + i*stride`` (negative ``i`` wraps).  Iterable and ``len()``-able."""

    def __init__(self, off: int, stride: int, length: int, struct_cls):
        self.off = off
        self.stride = stride
        self.length = length
        self.struct_cls = struct_cls

    def __get__(self, o, owner=None):
        if o is None:
            return self
        return _StructArrayView(o._backend, o._base + self.off, self.stride, self.length, self.struct_cls)


class _StructArrayView:
    __slots__ = ("_backend", "_base", "_stride", "length", "_cls")

    def __init__(self, backend, base: int, stride: int, length: int, cls):
        self._backend = backend
        self._base = base
        self._stride = stride
        self.length = length
        self._cls = cls

    def __getitem__(self, i: int):
        if i < 0:
            i += self.length
        return self._cls(self._backend, self._base + i * self._stride)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for i in range(self.length):
            yield self._cls(self._backend, self._base + i * self._stride)


# ---- view bases ---------------------------------------------------------------------------------------------

class StructView:
    """A view over ONE fixed-layout struct at a ``base`` address; its field descriptors add their own
    (relative) offset to ``base``.  Bind it to a backend + base — arrays hand it both."""

    __slots__ = ("_backend", "_base")

    def __init__(self, backend, base: int = 0):
        self._backend = backend
        self._base = base


def coerce_backend(source, base: int = 0):
    """A backend passes through; a live :class:`~c64_re.memory.Memory` (anything
    with a ``.ram`` bytearray) becomes a :class:`RamBackend`; anything else (a
    raw ``bytes``/``bytearray`` image) is wrapped in a :class:`ByteBackend` at
    ``base`` (the $XXXX address of the view's offset 0)."""
    if isinstance(source, (ByteBackend, RamBackend, OverlayBackend, WidthContractBackend)):
        return source
    if hasattr(source, "ram"):
        return RamBackend(source, base)
    return ByteBackend(source, base)
