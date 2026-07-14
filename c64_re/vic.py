"""VIC-II model: raster clock, IRQs, per-line register latching, renderer.

Timing model (PAL 6569 by default): 63 CPU cycles per raster line, 312
lines per frame, driven deterministically from CPU cycle telemetry via
:meth:`VIC.tick`.  Register state is latched per raster line as the frame
progresses, so raster-split effects (mid-frame color/mode/scroll changes,
sprite multiplexing at line granularity) render correctly even though the
model is line-based, not cycle-based.

Honest limits (docs/hardware_support.md):

- No badline/sprite DMA cycle stealing: every line costs 63 cycles.  Code
  that counts cycles against the raster (tight splits, FLD/FLI) will land
  on slightly different lines than real hardware — deterministic here, but
  not hardware-timing-exact.
- Within-line register changes take effect for the whole line.
- Sprite collisions ($D01E/$D01F) are computed lazily per frame from the
  latched line state when first read; collision *interrupts* ($D01A bits
  1-2) fail loud until a real game needs them.
- Light pen reads return a fixed neutral value.
"""
from __future__ import annotations

from typing import Callable

# The classic "Pepto" palette — presentation-only; the oracle compares
# indexed color numbers, never RGB.
PALETTE = [
    (0x00, 0x00, 0x00), (0xFF, 0xFF, 0xFF), (0x68, 0x37, 0x2B), (0x70, 0xA4, 0xB2),
    (0x6F, 0x3D, 0x86), (0x58, 0x8D, 0x43), (0x35, 0x28, 0x79), (0xB8, 0xC7, 0x6F),
    (0x6F, 0x4F, 0x25), (0x43, 0x39, 0x00), (0x9A, 0x67, 0x59), (0x44, 0x44, 0x44),
    (0x6C, 0x6C, 0x6C), (0x9A, 0xD2, 0x84), (0x6C, 0x5E, 0xB5), (0x95, 0x95, 0x95),
]

PAL_LINES = 312
PAL_CYCLES_PER_LINE = 63
FIRST_DISPLAY_LINE = 51   # raster line of text row 0, pixel row 0 (25-row mode)
DISPLAY_LINES = 200
DISPLAY_WIDTH = 320
NREGS = 0x2F


class VIC:
    def __init__(
        self,
        fetch: Callable[[int], int],
        color_ram: bytearray,
        *,
        lines: int = PAL_LINES,
        cycles_per_line: int = PAL_CYCLES_PER_LINE,
    ) -> None:
        self.fetch = fetch          # 14-bit VIC bus fetch (bank applied by machine)
        self.color_ram = color_ram
        self.lines = lines
        self.cycles_per_line = cycles_per_line
        self.regs = bytearray(NREGS)
        self.regs[0x11] = 0x1B      # power-on: display enabled, 25 rows, yscroll 3
        self.regs[0x16] = 0x08      # 40 columns
        self.regs[0x18] = 0x14      # screen $0400, chars $1000
        self.raster = 0
        self.raster_cmp = 0
        self.cycle_acc = 0
        self.frame = 0
        self.irq_status = 0         # $D019 low 4 bits
        self.irq_enable = 0         # $D01A low 4 bits
        self.sprite_sprite_coll = 0  # latched $D01E
        self.sprite_bg_coll = 0      # latched $D01F
        self._coll_frame = -1        # frame for which collisions were computed
        # per-line register latch for the frame being scanned
        self.line_regs: list[bytes] = [bytes(NREGS)] * lines
        self._latch_line()

    # ---- time --------------------------------------------------------------
    def _latch_line(self) -> None:
        self.line_regs[self.raster] = bytes(self.regs)

    def tick(self, cycles: int) -> None:
        self.cycle_acc += cycles
        while self.cycle_acc >= self.cycles_per_line:
            self.cycle_acc -= self.cycles_per_line
            self.raster += 1
            if self.raster >= self.lines:
                self.raster = 0
                self.frame += 1
            self._latch_line()
            if self.raster == self.raster_cmp:
                self.irq_status |= 0x01

    def irq_asserted(self) -> bool:
        return bool(self.irq_status & self.irq_enable & 0x0F)

    # ---- registers ------------------------------------------------------------
    def read(self, reg: int) -> int:
        reg &= 0x3F
        if reg >= NREGS:
            return 0xFF  # unmapped VIC registers read $FF
        if reg == 0x11:
            return (self.regs[0x11] & 0x7F) | ((self.raster >> 1) & 0x80)
        if reg == 0x12:
            return self.raster & 0xFF
        if reg == 0x13 or reg == 0x14:
            return 0  # light pen
        if reg == 0x16:
            return self.regs[0x16] | 0xC0
        if reg == 0x18:
            return self.regs[0x18] | 0x01
        if reg == 0x19:
            v = self.irq_status | 0x70
            if self.irq_status & self.irq_enable & 0x0F:
                v |= 0x80
            return v
        if reg == 0x1A:
            return self.irq_enable | 0xF0
        if reg == 0x1E:
            self._ensure_collisions()
            v = self.sprite_sprite_coll
            self.sprite_sprite_coll = 0
            return v
        if reg == 0x1F:
            self._ensure_collisions()
            v = self.sprite_bg_coll
            self.sprite_bg_coll = 0
            return v
        if 0x20 <= reg <= 0x2E:
            return self.regs[reg] | 0xF0
        return self.regs[reg]

    def write(self, reg: int, val: int) -> None:
        reg &= 0x3F
        val &= 0xFF
        if reg >= NREGS:
            return  # writes to unmapped registers are inert on real hardware
        if reg == 0x11:
            self.raster_cmp = (self.raster_cmp & 0xFF) | ((val & 0x80) << 1)
            self.regs[0x11] = val & 0x7F
        elif reg == 0x12:
            self.raster_cmp = (self.raster_cmp & 0x100) | val
        elif reg == 0x19:
            self.irq_status &= ~val & 0x0F  # write 1s to acknowledge
        elif reg == 0x1A:
            if val & 0x06:
                raise NotImplementedError(
                    "sprite collision interrupts enabled ($D01A bits 1-2) — "
                    "collision IRQ timing is not modeled yet; add it with an "
                    "observed contract when a real game needs it"
                )
            self.irq_enable = val & 0x0F
        elif reg in (0x1E, 0x1F):
            pass  # collision registers are read-only
        else:
            self.regs[reg] = val
        # keep the current line's latch coherent with late-in-line writes
        self.line_regs[self.raster] = bytes(self.regs)

    # ---- rendering -------------------------------------------------------------------
    def render_frame(self, *, border: bool = False):
        """Render the frame from the per-line latched registers + current memory.

        Returns ``(width, height, bytearray)`` of C64 color indices (0-15).
        With ``border=True`` the PAL-visible border is included (384x272).
        """
        bw = 32 if border else 0
        top = 36 if border else 0
        bottom = 36 if border else 0
        width = DISPLAY_WIDTH + 2 * bw
        height = DISPLAY_LINES + top + bottom
        out = bytearray(width * height)
        for y in range(-top, DISPLAY_LINES + bottom):
            line = FIRST_DISPLAY_LINE + y
            if not (0 <= line < self.lines):
                continue
            regs = self.line_regs[line]
            row_off = (y + top) * width
            border_color = regs[0x20] & 0x0F
            if border:
                for x in range(bw):
                    out[row_off + x] = border_color
                    out[row_off + width - 1 - x] = border_color
            if not (0 <= y < DISPLAY_LINES):
                for x in range(bw, width - bw):
                    out[row_off + x] = border_color
                continue
            pixels, fg = self._render_bg_line(y, regs)
            self._render_sprites_line(line, regs, pixels, fg)
            out[row_off + bw: row_off + bw + DISPLAY_WIDTH] = pixels
        return width, height, out

    def _render_bg_line(self, y: int, regs: bytes) -> tuple[bytearray, bytearray]:
        """One 320-pixel display line + its foreground mask (for priority
        and collisions).  ``y`` is 0-199 within the display window."""
        d011, d016, d018 = regs[0x11], regs[0x16], regs[0x18]
        bg0 = regs[0x21] & 0x0F
        pixels = bytearray([bg0]) * DISPLAY_WIDTH
        fg = bytearray(DISPLAY_WIDTH)
        border_color = regs[0x20] & 0x0F
        if not (d011 & 0x10):  # DEN off: blanked line shows border color
            for x in range(DISPLAY_WIDTH):
                pixels[x] = border_color
            return pixels, fg
        yscroll = d011 & 0x07
        xscroll = d016 & 0x07
        ecm = bool(d011 & 0x40)
        bmm = bool(d011 & 0x20)
        mcm = bool(d016 & 0x10)
        src_y = y - (yscroll - 3)
        if not (0 <= src_y < DISPLAY_LINES):
            return pixels, fg  # vertical idle area: background color
        row, pix_row = divmod(src_y, 8)
        screen_base = ((d018 >> 4) & 0x0F) * 0x400
        char_base = ((d018 >> 1) & 0x07) * 0x800
        bitmap_base = 0x2000 if (d018 & 0x08) else 0x0000
        fetch = self.fetch
        cram = self.color_ram
        bgs = (regs[0x21] & 0x0F, regs[0x22] & 0x0F, regs[0x23] & 0x0F, regs[0x24] & 0x0F)

        for col in range(40):
            cell = row * 40 + col
            sc = fetch(screen_base + cell)
            color = cram[cell] & 0x0F
            if bmm:
                data = fetch(bitmap_base + row * 320 + col * 8 + pix_row)
                if mcm:
                    colors = (bgs[0], (sc >> 4) & 0x0F, sc & 0x0F, color)
                    self._put_mc(pixels, fg, col, xscroll, data, colors)
                else:
                    self._put_hires(pixels, fg, col, xscroll, data,
                                    sc & 0x0F, (sc >> 4) & 0x0F)
            else:
                if ecm:
                    char_index = sc & 0x3F
                    bg = bgs[(sc >> 6) & 0x03]
                    data = fetch(char_base + char_index * 8 + pix_row)
                    self._put_hires(pixels, fg, col, xscroll, data, bg, color)
                elif mcm and (color & 0x08):
                    data = fetch(char_base + sc * 8 + pix_row)
                    colors = (bgs[0], bgs[1], bgs[2], color & 0x07)
                    self._put_mc(pixels, fg, col, xscroll, data, colors)
                else:
                    data = fetch(char_base + sc * 8 + pix_row)
                    self._put_hires(pixels, fg, col, xscroll, data, bgs[0], color)
        if xscroll:
            for x in range(xscroll):
                pixels[x] = bgs[0]
                fg[x] = 0
        if not (d016 & 0x08):  # 38-column mode: side borders overlay the edges
            for x in range(7):
                pixels[x] = border_color
                fg[x] = 0
            for x in range(DISPLAY_WIDTH - 9, DISPLAY_WIDTH):
                pixels[x] = border_color
                fg[x] = 0
        if not (d011 & 0x08):  # 24-row mode: top/bottom rows show border
            if y < 4 or y >= DISPLAY_LINES - 4:
                for x in range(DISPLAY_WIDTH):
                    pixels[x] = border_color
                    fg[x] = 0
        return pixels, fg

    @staticmethod
    def _put_hires(pixels, fg, col, xscroll, data, bg, fgcolor) -> None:
        base = col * 8 + xscroll
        for b in range(8):
            x = base + b
            if x >= DISPLAY_WIDTH:
                break
            if (data >> (7 - b)) & 1:
                pixels[x] = fgcolor
                fg[x] = 1
            else:
                pixels[x] = bg

    @staticmethod
    def _put_mc(pixels, fg, col, xscroll, data, colors) -> None:
        base = col * 8 + xscroll
        for p in range(4):
            pair = (data >> (6 - 2 * p)) & 0x03
            color = colors[pair]
            is_fg = 1 if pair >= 2 else 0
            for sub in range(2):
                x = base + p * 2 + sub
                if x >= DISPLAY_WIDTH:
                    break
                pixels[x] = color
                fg[x] = is_fg

    # ---- sprites --------------------------------------------------------------------
    def _sprite_line_pixels(self, s: int, line: int, regs: bytes):
        """Sprite ``s``'s pixels on raster ``line`` as a list of
        (x_on_screen, color_index) — or None if it doesn't cover the line."""
        if not (regs[0x15] >> s) & 1:
            return None
        sy = regs[2 * s + 1]
        yexp = (regs[0x17] >> s) & 1
        height = 42 if yexp else 21
        if not (sy <= line < sy + height):
            return None
        row = (line - sy) >> (1 if yexp else 0)
        d018 = regs[0x18]
        screen_base = ((d018 >> 4) & 0x0F) * 0x400
        ptr = self.fetch(screen_base + 0x3F8 + s)
        data = [self.fetch(ptr * 64 + row * 3 + i) for i in range(3)]
        sx = regs[2 * s] | (((regs[0x10] >> s) & 1) << 8)
        xexp = (regs[0x1D] >> s) & 1
        mc = (regs[0x1C] >> s) & 1
        color = regs[0x27 + s] & 0x0F
        mc0 = regs[0x25] & 0x0F
        mc1 = regs[0x26] & 0x0F
        pixels: list[tuple[int, int]] = []
        x0 = sx - 24  # sprite x 24 == leftmost display pixel
        if mc:
            for p in range(12):
                byte = data[p // 4]
                pair = (byte >> (6 - 2 * (p % 4))) & 0x03
                if pair == 0:
                    continue
                c = (mc0, color, mc1)[pair - 1]
                w = 4 if xexp else 2
                for sub in range(w):
                    x = x0 + p * w + sub
                    if 0 <= x < DISPLAY_WIDTH:
                        pixels.append((x, c))
        else:
            for p in range(24):
                if not (data[p // 8] >> (7 - (p % 8))) & 1:
                    continue
                w = 2 if xexp else 1
                for sub in range(w):
                    x = x0 + p * w + sub
                    if 0 <= x < DISPLAY_WIDTH:
                        pixels.append((x, color))
        return pixels

    def _render_sprites_line(self, line: int, regs: bytes,
                             pixels: bytearray | None, fg: bytearray) -> tuple[int, int]:
        """Draw sprites onto a rendered line (respecting bg priority) and
        collect collision bits; pass ``pixels=None`` for collision-only.
        Returns (sprite_sprite_bits, sprite_bg_bits)."""
        occupancy = {}
        ss_bits = 0
        sb_bits = 0
        priority = regs[0x1B]
        for s in range(7, -1, -1):  # draw 7 first so 0 ends up on top
            spx = self._sprite_line_pixels(s, line, regs)
            if not spx:
                continue
            behind = (priority >> s) & 1
            for x, c in spx:
                prev = occupancy.get(x)
                if prev is not None:
                    ss_bits |= (1 << s) | (1 << prev)
                occupancy[x] = s
                if fg[x]:
                    sb_bits |= 1 << s
                if pixels is not None and not (behind and fg[x]):
                    pixels[x] = c
        return ss_bits, sb_bits

    def _ensure_collisions(self) -> None:
        """Lazily compute this frame's sprite collisions from the latched
        line state (once per frame; results OR into the latched registers)."""
        if self._coll_frame == self.frame:
            return
        self._coll_frame = self.frame
        ss_total = 0
        sb_total = 0
        for y in range(DISPLAY_LINES):
            line = FIRST_DISPLAY_LINE + y
            regs = self.line_regs[line]
            if not regs[0x15]:
                continue
            _, fg = self._render_bg_line(y, regs)
            ss, sb = self._render_sprites_line(line, regs, None, fg)
            ss_total |= ss
            sb_total |= sb
        self.sprite_sprite_coll |= ss_total
        self.sprite_bg_coll |= sb_total

    # ---- snapshot ----------------------------------------------------------------
    def get_state(self) -> dict:
        return {
            "regs": bytes(self.regs),
            "raster": self.raster,
            "raster_cmp": self.raster_cmp,
            "cycle_acc": self.cycle_acc,
            "frame": self.frame,
            "irq_status": self.irq_status,
            "irq_enable": self.irq_enable,
            "sprite_sprite_coll": self.sprite_sprite_coll,
            "sprite_bg_coll": self.sprite_bg_coll,
        }

    def set_state(self, state: dict) -> None:
        self.regs[:] = state["regs"]
        self.raster = state["raster"]
        self.raster_cmp = state["raster_cmp"]
        self.cycle_acc = state["cycle_acc"]
        self.frame = state["frame"]
        self.irq_status = state["irq_status"]
        self.irq_enable = state["irq_enable"]
        self.sprite_sprite_coll = state["sprite_sprite_coll"]
        self.sprite_bg_coll = state["sprite_bg_coll"]
        self._coll_frame = -1
        self.line_regs = [bytes(self.regs)] * self.lines
