"""Lettering and Christmas decorations laid over a picture.

A picture on the ornament is a map of wall thickness: thin glows, thick stays
dark.  Lettering and decorations are drawn in those same terms - "light" is
the thinnest wall, "dark" the thickest, "plain" the frame thickness - on a
flat sheet that is then laid over the photo (or over a plain panel, when a
picture has no photo and is only lettering).

Everything is drawn where it will be ON THE SPHERE: every shape is given in
millimetres on the surface and is carried through the same projection that the
sheet is later read back with, so a round bauble comes out round and a 1 mm
stroke 1 mm wide wherever it sits.  Shapes are bold on purpose: a printed line
is only as good as the nozzle that lays it, and near the top of a picture the
printer cannot build a sharp edge on the outside at all (imaging.outside_first),
so only shapes a few millimetres across survive there.

Every shape carries its own contrast (light art with a dark edge, or the other
way round), because it has to read on any photo and on any plain panel.

The plain parts of the ball itself - the dividers between the pictures, the
strips above and below them, the closed end - are decorated with the same
tools; see "the plain parts of the ball" at the end of this module.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
from scipy import ndimage

from . import fonts
from .imaging import panel_uv, window_extent, wrap_angle
from .params import BACK_LON, ImageSlot, Layout, Params

SUPERSAMPLE = 3.0           # sheet pixels per sample of the model
MAX_SHEET_PX = 4096         # longest side of a sheet
EDGE_MM = 0.6               # contrasting edge drawn round every decoration
MARGIN_MM = 1.2             # lettering keeps this far from the edge of the picture
LINE_PITCH = 0.92           # line spacing, as a share of the font's own line height
MAX_LINES = 6
_REF_PX = 160               # font size at which lettering is measured

# (hidden, light, plain) - what a shape of each tone does to the picture underneath:
# how much of it is hidden, and how much of what hides it is light or plain frame.
_FILL = {"light": (255, 255, 0), "dark": (255, 0, 0), "plain": (255, 0, 255), "clear": (0, 0, 0)}
_OTHER = {"light": "dark", "dark": "light"}


@dataclass(frozen=True)
class Panel:
    """Size of one picture on the sphere."""

    radius: float           # mm
    panel_half: float       # half its width, radians of longitude
    band_half: float        # half its height, radians of latitude

    @property
    def projection(self) -> str:
        """How the flat sheet is laid on the sphere: true to shape about the
        middle of the picture, or longitude / latitude for a picture too wide
        for that (one that goes all the way round)."""
        return "natural" if self.panel_half <= math.radians(100.0) else "wrap"

    @staticmethod
    def of(lay: Layout) -> "Panel":
        return Panel(lay.R, lay.panel_half, lay.band_half)


def offset(dlam, phi, east, north, radius: float):
    """The sphere points `east` and `north` millimetres from the points
    (dlam, phi), measured along the surface (exact: an azimuthal-equidistant
    step).  Everything broadcasts."""
    east, north = np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)
    rho = np.hypot(east, north)
    c = rho / radius
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(rho > 1e-9, np.sin(c) / np.maximum(rho, 1e-12), 1.0 / radius)
    sin0, cos0 = np.sin(phi), np.cos(phi)
    lat = np.arcsin(np.clip(np.cos(c) * sin0 + north * s * cos0, -1.0, 1.0))
    lon = dlam + np.arctan2(east * s, cos0 * np.cos(c) - north * s * sin0)
    return lon, lat


def flat_uv(dlam, phi, projection: str):
    """A sphere point on a flat sheet, in radians of arc: imaging.panel_uv for
    a picture, and "polar" - true to shape about the south pole, the closed
    end of the ornament - for what is drawn there."""
    if projection == "polar":
        c = np.asarray(phi, dtype=np.float64) + math.pi / 2
        return c * np.sin(dlam), -c * np.cos(dlam)
    return panel_uv(np.asarray(dlam, dtype=np.float64), np.asarray(phi, dtype=np.float64), projection)


class Sheet:
    """The flat drawing surface of one picture - or of a plain part of the
    ball (see ball_layers).  Shapes are given on the sphere.

    window: (u_max, v_top, v_bottom) in radians, for a sheet that is not a
    whole picture centred on the equator; projection: how it is laid on the
    sphere, if not the way the panel's own picture would be."""

    def __init__(self, panel: Panel, px_per_mm: float, projection: str | None = None,
                 window: tuple[float, float, float] | None = None):
        self.panel = panel
        self.projection = projection or panel.projection
        if window is None:
            self.u_max, self.v_max = window_extent(panel.panel_half, panel.band_half, self.projection)
            v_bottom = -self.v_max
        else:
            self.u_max, self.v_max, v_bottom = window        # (v_max is the top edge of the sheet)
        longest = max(2 * self.u_max, self.v_max - v_bottom) * panel.radius * px_per_mm
        if longest > MAX_SHEET_PX:
            px_per_mm *= MAX_SHEET_PX / longest
        self.px_per_mm = px_per_mm
        self.k = panel.radius * px_per_mm                   # pixels per radian of arc
        self.size = (int(math.ceil(2 * self.u_max * self.k)) + 3,
                     int(math.ceil((self.v_max - v_bottom) * self.k)) + 3)
        self.image = Image.new("RGB", self.size, _FILL["clear"])
        self.draw = ImageDraw.Draw(self.image)
        self.swap = False           # True: "light" is drawn dark and "dark" light
        self._grouped = False       # True while the pieces of one shape are being collected

    def _fill(self, tone: str):
        if self._grouped:
            return 255
        return _FILL[_OTHER[tone] if self.swap and tone in _OTHER else tone]

    @contextmanager
    def group(self, tone: str, edge: float = 0.0):
        """Everything drawn inside the block is ONE shape of this tone: its
        pieces run together, and a contrasting edge `edge` mm wide goes round
        the whole of it instead of round every piece.  (Inside the block the
        tone and edge of each piece are ignored.)"""
        image, draw = self.image, self.draw
        mask = Image.new("L", self.size, 0)
        self.image, self.draw, self._grouped = mask, ImageDraw.Draw(mask), True
        try:
            yield
        finally:
            self.image, self.draw, self._grouped = image, draw, False
        box = mask.getbbox()
        if box is None:
            return
        if edge > 0.0 and tone in _OTHER:
            reach = edge * self.px_per_mm
            grow = int(math.ceil(reach)) + 2
            box = (max(0, box[0] - grow), max(0, box[1] - grow),
                   min(self.size[0], box[2] + grow), min(self.size[1], box[3] + grow))
            inked = np.asarray(mask.crop(box)) > 127
            near = ndimage.distance_transform_edt(~inked) <= reach
            self.image.paste(self._fill(_OTHER[tone]), box,
                             Image.fromarray(near.astype(np.uint8) * 255))
        self.image.paste(self._fill(tone), (0, 0), mask)

    # -- where things are -------------------------------------------------------
    def to_px(self, dlam, phi):
        u, v = flat_uv(dlam, phi, self.projection)
        return (u + self.u_max) * self.k + 1.0, (self.v_max - v) * self.k + 1.0

    def offset(self, dlam, phi, east, north):
        return offset(dlam, phi, east, north, self.panel.radius)

    # -- shapes -------------------------------------------------------------------
    def fill(self, dlam, phi, tone: str) -> None:
        """Fill the polygon whose corners are the given sphere points."""
        x, y = self.to_px(dlam, phi)
        if len(x) >= 3:
            self.draw.polygon(np.column_stack((x, y)).ravel().tolist(), fill=self._fill(tone))

    def shape(self, at: tuple[float, float], east, north, tone: str, edge: float = 0.0) -> None:
        """Fill a polygon given in millimetres east / north of the sphere point
        `at`.  edge > 0 first draws a contrasting line that wide round it."""
        lon, lat = self.offset(at[0], at[1], east, north)
        if edge > 0.0:
            self.ribbon(lon, lat, 2 * edge, _OTHER[tone], closed=True)
        self.fill(lon, lat, tone)

    def shapes(self, dlam, phi, east, north, tone: str) -> None:
        """Many polygons at once.  east / north: (polygons, corners) in mm from
        the sphere points dlam / phi (one per polygon, or one for all)."""
        east, north = np.atleast_2d(east), np.atleast_2d(north)
        dlam = np.broadcast_to(np.asarray(dlam, dtype=np.float64), (east.shape[0],))[:, None]
        phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), (east.shape[0],))[:, None]
        x, y = self.to_px(*self.offset(dlam, phi, east, north))
        fill = self._fill(tone)
        for row in np.stack([x, y], axis=-1).reshape(east.shape[0], -1).tolist():
            self.draw.polygon(row, fill=fill)

    def disc(self, at: tuple[float, float], radius: float, tone: str, edge: float = 0.0) -> None:
        turn = np.linspace(0.0, 2 * math.pi, 28, endpoint=False)
        if edge > 0.0:
            self.fill(*self.offset(at[0], at[1], (radius + edge) * np.cos(turn),
                                   (radius + edge) * np.sin(turn)), _OTHER[tone])
        self.fill(*self.offset(at[0], at[1], radius * np.cos(turn), radius * np.sin(turn)), tone)

    def ribbon(self, dlam, phi, width, tone: str, closed: bool = False) -> None:
        """A band `width` mm wide along a path of sphere points (width may vary
        along it).  An open band gets round ends, and so does every sharp bend.

        The band is laid piece by piece, one for each step of the path.  Drawn
        as a single outline it folded over itself wherever the path turned
        more sharply than the band is wide - the corners of a zigzag, the
        point where two swags of a garland meet - and left a hole there."""
        dlam, phi = np.asarray(dlam, dtype=np.float64), np.asarray(phi, dtype=np.float64)
        n = len(dlam)
        if n < 2:
            return
        radius = self.panel.radius
        half = np.broadcast_to(np.asarray(width, dtype=np.float64) / 2, dlam.shape)
        a = np.arange(n if closed else n - 1)               # each piece runs from point a to point b
        b = (a + 1) % n
        d_e = (dlam[b] - dlam[a]) * np.cos(0.5 * (phi[a] + phi[b])) * radius
        d_n = (phi[b] - phi[a]) * radius
        length = np.hypot(d_e, d_n)
        real = length > 1e-9                                # (a point given twice is no piece)
        a, b = a[real], b[real]
        if len(a) == 0:
            return
        # the left-hand normal of each piece, east and north
        own = np.stack([-d_n[real] / length[real], d_e[real] / length[real]], axis=1)
        # Where two pieces meet at a gentle bend they share a normal (a mitre), so that the
        # band has no notch; at a sharp bend each keeps its own and a disc rounds the corner.
        after = np.roll(own, -1, axis=0)                    # the piece that follows each piece
        both = own + after
        gentle = (own * after).sum(axis=1) >= math.cos(math.radians(25.0))
        if not closed:
            gentle[-1] = False                              # nothing follows the last piece
        with np.errstate(divide="ignore", invalid="ignore"):
            mitre = 2.0 * both / np.maximum((both * both).sum(axis=1, keepdims=True), 1e-12)
        at_end = np.where(gentle[:, None], mitre, own)
        at_start = np.where(np.roll(gentle, 1)[:, None], np.roll(mitre, 1, axis=0), own)
        corners = []
        for point, normal, side in ((a, at_start, 1.0), (b, at_end, 1.0), (b, at_end, -1.0),
                                    (a, at_start, -1.0)):
            reach = side * half[point]
            corners.append(self.to_px(*self.offset(dlam[point], phi[point], normal[:, 0] * reach,
                                                   normal[:, 1] * reach)))
        fill = self._fill(tone)
        quads = np.stack([c for corner in corners for c in corner], axis=1)
        for quad in quads.tolist():
            self.draw.polygon(quad, fill=fill)
        # round what the pieces leave open: sharp bends, and the two ends of an open band
        bends = b[~gentle] if closed else np.concatenate([b[~gentle][:-1], [a[0], b[-1]]])
        bends = bends[half[bends] * self.px_per_mm > 0.8]
        if len(bends):
            turn = np.linspace(0.0, 2 * math.pi, 20, endpoint=False)[None, :]
            x, y = self.to_px(*self.offset(dlam[bends][:, None], phi[bends][:, None],
                                           half[bends][:, None] * np.cos(turn),
                                           half[bends][:, None] * np.sin(turn)))
            for disc in np.stack([x, y], axis=-1).reshape(len(bends), -1).tolist():
                self.draw.polygon(disc, fill=fill)

    def needles(self, dlam, phi, angle, length, width, tone: str, edge: float = 0.0) -> None:
        """Many fir needles at once: like leaves(), but each is widest near
        its root and tapers in a long straight line to its point."""
        t = np.linspace(0.0, 1.0, 9)
        self.leaves(dlam, phi, angle, length, width, tone, edge,
                    belly=np.minimum(1.0, (t / 0.14) ** 0.6) * (1.0 - t ** 1.7) ** 0.85)

    def leaves(self, dlam, phi, angle, length, width, tone: str, edge: float = 0.0,
               belly: np.ndarray | None = None) -> None:
        """Many pointed leaves at once: each grows from its own sphere point,
        `length` mm long at `angle` radians from east and `width` mm across.
        belly: half the outline, as shares of the width at 9 even steps from
        root to tip (default: a pointed oval)."""
        dlam, phi, angle, length, width = (np.atleast_1d(np.asarray(a, dtype=np.float64))
                                           for a in (dlam, phi, angle, length, width))
        t = np.linspace(0.0, 1.0, 9)
        if belly is None:
            belly = np.sin(math.pi * t) ** 0.8
        way = np.concatenate([t, t[-2:0:-1]])[None, :]
        bulge = np.concatenate([belly, -belly[-2:0:-1]])[None, :]
        ca, sa = np.cos(angle)[:, None], np.sin(angle)[:, None]
        for grow, shade in ((edge, _OTHER[tone]), (0.0, tone)):
            if shade != tone and edge <= 0.0:
                continue
            along = way * (length[:, None] + 2 * grow) - grow
            across = bulge * (width[:, None] / 2 + grow)
            self.shapes(dlam, phi, along * ca - across * sa, along * sa + across * ca, shade)


# ---------------------------------------------------------------------------
# lettering
# ---------------------------------------------------------------------------
def text_lines(text: str) -> list[str]:
    """The lines of a slot's text as they are set: no blank lines at either
    end, no stray spaces (a tab or a thin space pasted in counts as one plain
    space - fonts have no shape for those), at most MAX_LINES."""
    lines = [" ".join(line.split()) for line in str(text or "").replace("\r", "").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines[:MAX_LINES]


def _ink_box(font, line: str) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of what a line of text really inks, in pixels
    from the start of its baseline, or None if it inks nothing.

    Not font.getbbox(): that box always takes in the room the font sets aside
    for the line, and an italic or a script leans well out of that room on one
    side and stops short of it on the other."""
    mask, (x, y) = font.getmask2(line, mode="L", anchor="ls")
    box = mask.getbbox()
    if box is None:
        return None
    return x + box[0], y + box[1], x + box[2], y + box[3]


@lru_cache(maxsize=256)
def _measure(font_name: str, lines: tuple[str, ...]) -> dict:
    """Lettering measured in units of the font size (1.0 = one em)."""
    font = fonts.load(font_name, _REF_PX)
    ascent, descent = font.getmetrics()
    pitch = LINE_PITCH * (ascent + descent)
    boxes = [(_ink_box(font, line) if line else None) or (0, 0, 0, 0) for line in lines]
    inked = [k for k, box in enumerate(boxes) if box != (0, 0, 0, 0)]
    top = -boxes[inked[0]][1] if inked else 0.0             # ink above the first baseline
    bottom = boxes[inked[-1]][3] if inked else 0.0          # ink below the last one
    width = max((box[2] - box[0] for box in boxes), default=0.0)
    cap_box = _ink_box(font, "H")
    cap = float(-cap_box[1]) if cap_box and cap_box[1] < 0 else 0.7 * _REF_PX
    return {"width": width / _REF_PX, "height": ((len(lines) - 1) * pitch + top + bottom) / _REF_PX,
            "top": top / _REF_PX, "pitch": pitch / _REF_PX, "cap": cap / _REF_PX,
            "stroke": _stroke(font_name, " ".join(lines)[:48]) / _REF_PX}


def _stroke(font_name: str, sample: str) -> float:
    """Typical width of the strokes of `sample`, in pixels at _REF_PX.

    Across a stroke the distance to its nearest edge rises from nothing to half
    the stroke's width and falls again, so it averages a quarter of the width."""
    font = fonts.load(font_name, _REF_PX)
    sample = sample.strip() or "Abc"
    box = font.getbbox(sample)
    image = Image.new("L", (max(4, box[2] - box[0] + 8), max(4, box[3] - box[1] + 8)), 0)
    ImageDraw.Draw(image).text((4 - box[0], 4 - box[1]), sample, font=font, fill=255)
    ink = np.asarray(image) > 127
    if not ink.any():
        return 0.1 * _REF_PX
    return float(4.0 * ndimage.distance_transform_edt(ink)[ink].mean())


@lru_cache(maxsize=256)
def missing_characters(font_name: str, text: str) -> str:
    """The characters of `text` that the font has no shape for (it would print
    an empty box in their place): a heart or a star in a font of plain letters."""
    font = fonts.load(font_name, 48)
    try:
        blank = font.getmask("\uffff")
        box = (blank.size, bytes(blank))
    except Exception:           # a font that cannot even draw its stand-in: nothing to compare
        return ""
    missing = []
    for char in sorted(set(text)):
        if char.isspace():
            continue
        try:
            mask = font.getmask(char)
        except Exception:
            missing.append(char)
            continue
        if mask.getbbox() is None or (mask.size, bytes(mask)) == box:
            missing.append(char)
    return "".join(missing)


def stroke_share(font_name: str, sample: str = "Happy Holidays") -> float:
    """Typical stroke width of a font as a share of the height of its capitals."""
    m = _measure(fonts.resolve(font_name), (sample,))
    return m["stroke"] / m["cap"]


@dataclass(frozen=True)
class Lettering:
    """Where a slot's text goes, and how big it comes out."""

    lines: tuple[str, ...]
    font: str               # the font really used
    substituted: bool       # True if the font asked for is not on this computer
    size: float             # font size, mm
    u: float                # centre of the block on the flat sheet, mm right of the middle ...
    v: float                # ... and mm above it
    width: float            # the block with its outline, mm
    height: float
    cap: float              # height of the capital letters, mm
    stroke: float           # typical width of a stroke, mm
    stretch: float          # widening that makes up for the sheet's own squeeze ("wrap" only)
    radius: float           # of the sphere, mm
    missing: str = ""       # characters the font has no shape for

    @property
    def latitude(self) -> float:
        """Latitude of the middle of the block, radians."""
        return self.v / self.radius

    @property
    def top_latitude(self) -> float:
        return (self.v + self.height / 2) / self.radius


def _half_width_at(panel: Panel):
    """Function giving half the width of the picture, in mm, at a height of v
    mm above or below its middle (the narrowest it gets up to there)."""
    radius = panel.radius
    if panel.projection == "wrap":
        return lambda v: panel.panel_half * radius * math.cos(min(abs(v) / radius, panel.band_half))
    phi = np.linspace(0.0, panel.band_half, 49)
    u, v = panel_uv(np.full_like(phi, panel.panel_half), phi, "natural")
    narrowest = np.minimum.accumulate(u) * radius
    heights = v * radius
    return lambda at: float(np.interp(abs(at), heights, narrowest))


# How far each decoration reaches in from its edge of the picture, in units of
# deco_unit() - the strip the lettering keeps out of.  (Holly sits in the
# corners only and a wreath goes round the middle: neither takes a strip.)
_REACH_TOP = {"garland": 16.0, "lights": 13.4, "icicles": 13.5}
_REACH_BOTTOM = {"garland": 16.0, "snow": 9.0, "trees": 12.0}
_REACH_FRAME = {"line": 2.9, "candy": 6.6, "wreath_square": 11.4}


def reserved(slot: ImageSlot, panel: Panel) -> tuple[float, float, float]:
    """(top, bottom, sides): the depth in mm of the strips along the edges of a
    picture that its decorations take up."""
    s = deco_unit(slot, panel)
    frame = _REACH_FRAME.get(slot.deco_frame, 0.0) * frame_unit(slot, panel)
    return (max(frame, _REACH_TOP.get(slot.deco_top, 0.0) * s),
            max(frame, _REACH_BOTTOM.get(slot.deco_bottom, 0.0) * s), frame)


def lettering_for(slot: ImageSlot, panel: Panel) -> Lettering | None:
    """Lay out a slot's text on its picture, or None if it has none.

    The size setting is a share of the largest size at which the whole block
    still fits the room it has: the picture, less the strips along the edges
    that a garland or a border takes up.  Nearer the top or the bottom the
    picture is narrower, and a block moved there is made smaller if it has to
    be: lettering is never cut off at the edge."""
    lines = tuple(text_lines(slot.text))
    if not any(lines):
        return None
    asked = slot.text_font or fonts.default_font()
    font_name = fonts.resolve(asked)
    m = _measure(font_name, lines)
    if m["width"] <= 0.0:
        return None
    extra = max(0.0, slot.text_bold) + max(0.0, slot.text_outline)     # round the ink, each side
    radius = panel.radius
    half_width = _half_width_at(panel)
    top, bottom, sides = reserved(slot, panel)
    edge = panel.band_half * radius - MARGIN_MM
    v_hi, v_lo = edge - top, -(edge - bottom)                  # the room between the strips
    ring = None
    if slot.deco_frame == "wreath":         # a round wreath: the lettering goes inside the ring
        a, b, bough = wreath_axes(panel, frame_unit(slot, panel))
        ring = (max(2.0, a - bough - MARGIN_MM), max(2.0, b - bough - MARGIN_MM))
        v_hi, v_lo = min(v_hi, ring[1]), max(v_lo, -ring[1])
    if v_hi - v_lo < 0.25 * edge:                              # decorations all but fill it:
        v_hi, v_lo = edge - sides, -(edge - sides)             # then the lettering goes over them
        if ring is not None:
            v_hi, v_lo = ring[1], -ring[1]
    room_v = (v_hi - v_lo) / 2

    def room_u(v: float, half_h: float) -> float:
        far = max(abs(v - half_h), abs(v + half_h))
        room = half_width(far) - MARGIN_MM - sides
        if ring is not None:                # as wide as the oval is at the block's far edge
            room = min(room, ring[0] * math.sqrt(max(0.0, 1.0 - (far / ring[1]) ** 2)))
        return room

    def span(half_w: float, half_h: float) -> tuple[float, float]:
        """(lowest, highest) the middle of a block this size may sit: between
        the strips the decorations take up, and - inside a round wreath - no
        further from the middle of the ring than where its corners reach it.
        (An oval closes to nothing at its top and bottom: a block that followed
        it there would have to shrink to a speck.)"""
        low, high = v_lo + half_h, v_hi - half_h
        if ring is not None:
            reach = ring[1] * math.sqrt(max(0.0, 1.0 - (half_w / ring[0]) ** 2)) - half_h
            low, high = max(low, -reach), min(high, reach)
        if high < low:                      # no room to move at all: where it fits best
            floor, ceiling = v_lo + half_h, v_hi - half_h
            want = 0.0 if ring is not None else (v_lo + v_hi) / 2
            low = high = min(max(want, floor), ceiling) if floor <= ceiling else (v_lo + v_hi) / 2
        return low, high

    def fits(size: float) -> bool:
        half_h, half_w = m["height"] * size / 2 + extra, m["width"] * size / 2 + extra
        if half_h > room_v:
            return False
        # the widest place the block is allowed to be: the middle of the ring, or of the room
        best = 0.0 if ring is not None else (v_lo + v_hi) / 2
        best = min(max(best, v_lo + half_h), v_hi - half_h)
        return half_w <= room_u(best, half_h)

    lo, hi = 0.0, 4.0 * radius
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            lo = mid
        else:
            hi = mid
    size = max(0.5, lo * min(1.0, max(0.1, slot.text_size)))
    v = wide = half_h = half_w = 0.0
    for _ in range(6):
        half_h = m["height"] * size / 2 + extra
        half_w = m["width"] * size / 2 + extra
        low, high = span(half_w, half_h)
        v = (low + high) / 2 + slot.text_y * (high - low) / 2
        wide = room_u(v, half_h)
        if half_w <= wide + 1e-6 or size <= 0.5:
            break
        size = max(0.5, size * max(0.2, (wide - extra) / max(1e-6, half_w - extra)) * 0.995)
    room_u = wide
    stretch = 1.0 / math.cos(v / radius) if panel.projection == "wrap" else 1.0
    u = slot.text_x * max(0.0, room_u - half_w) * stretch
    return Lettering(lines=lines, font=font_name, substituted=font_name != asked,
                     size=size, u=u, v=v, width=2 * half_w, height=2 * half_h,
                     cap=m["cap"] * size, stroke=m["stroke"] * size + 2 * max(0.0, slot.text_bold),
                     stretch=stretch, radius=radius,
                     missing=missing_characters(font_name, "".join(lines)))


def _draw_lettering(sheet: Sheet, slot: ImageSlot, text: Lettering) -> None:
    scale = sheet.px_per_mm
    m = _measure(text.font, text.lines)
    size_px = text.size * scale                         # the size the block was laid out for
    # The block was measured at a large size.  Small sizes are fitted to the pixel grid and
    # can come out a little wider: step down until the widest line is inside its block again.
    px = max(4, int(round(size_px)))
    while True:
        font = fonts.load(text.font, px)
        boxes = [_ink_box(font, line) if line else None for line in text.lines]
        widest = max((box[2] - box[0] for box in boxes if box), default=0)
        if widest <= m["width"] * size_px + 1.0 or px <= 4:
            break
        px -= 1
    bold = int(round(max(0.0, slot.text_bold) * scale))
    outline = int(round(max(0.0, slot.text_outline) * scale))
    tone = slot.text_tone if slot.text_tone in _OTHER else "light"
    pad = bold + outline + 3
    width = int(math.ceil(m["width"] * size_px)) + 2 * pad + int(0.5 * size_px)
    height = int(math.ceil(m["height"] * size_px)) + 2 * pad + int(0.5 * size_px)
    block = Image.new("RGB", (width, height), _FILL["clear"])
    draw = ImageDraw.Draw(block)
    first = (height - m["height"] * size_px) / 2 + m["top"] * size_px
    passes = [(bold, _FILL[tone])]
    if outline > 0:
        passes.insert(0, (bold + outline, _FILL[_OTHER[tone]]))
    for grow, fill in passes:           # every outline first, so none covers a neighbour's letters
        for k, line in enumerate(text.lines):
            if line and boxes[k]:
                # centred on its ink, not on the room the font sets aside for it: an italic
                # or a script leans well out of that
                left = width / 2 - (boxes[k][0] + boxes[k][2]) / 2
                draw.text((left, first + k * m["pitch"] * size_px), line, font=font, fill=fill,
                          anchor="ls", stroke_width=grow, stroke_fill=fill)
    if abs(text.stretch - 1.0) > 1e-3:
        width = max(2, int(round(width * text.stretch)))
        block = block.resize((width, height), Image.Resampling.BICUBIC)
    x = int(round((text.u / sheet.panel.radius + sheet.u_max) * sheet.k + 1.0 - width / 2))
    y = int(round((sheet.v_max - text.v / sheet.panel.radius) * sheet.k + 1.0 - height / 2))
    # lay the block over the sheet: what it hides is hidden, the rest shows through
    box = (max(0, x), max(0, y), min(sheet.size[0], x + width), min(sheet.size[1], y + height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    block = block.crop((box[0] - x, box[1] - y, box[2] - x, box[3] - y))
    under = sheet.image.crop(box)
    keep = ImageChops.invert(block.getchannel(0))
    under = ImageChops.multiply(under, Image.merge("RGB", (keep, keep, keep)))
    sheet.image.paste(ImageChops.add(under, block), box)


# ---------------------------------------------------------------------------
# decorations
# ---------------------------------------------------------------------------
@dataclass
class _Claims:
    """The parts of a picture that are spoken for, so that scattered shapes
    keep off them.  All in millimetres."""

    top: float = 0.0            # a strip this deep along the top edge
    bottom: float = 0.0
    sides: float = 0.0          # a border this wide all the way round
    corners: list[tuple[bool, float, float]] = field(default_factory=list)  # (top?, deep, wide)
    ring: tuple[float, float] | None = None         # everything inside this oval (half-axes)
    block: tuple[float, float, float, float] | None = None   # lettering: u, v, half width, half height
    middle: bool = False        # the middle of a photo, where the faces are

    def free(self, panel: Panel, dlam: np.ndarray, phi: np.ndarray, gap: float) -> np.ndarray:
        """Which of the given points are at least `gap` mm clear of everything."""
        radius = panel.radius
        from_top = (panel.band_half - phi) * radius
        from_bottom = (phi + panel.band_half) * radius
        from_side = (panel.panel_half - np.abs(dlam)) * radius * np.cos(phi)
        ok = ((from_top > self.top + gap) & (from_bottom > self.bottom + gap)
              & (np.minimum(from_side, np.minimum(from_top, from_bottom)) > self.sides + gap))
        for is_top, deep, wide in self.corners:
            ok &= ~(((from_top if is_top else from_bottom) < deep + gap) & (from_side < wide + gap))
        u, v = panel_uv(dlam, phi, panel.projection)
        u, v = u * radius, v * radius
        if self.ring is not None:
            ok &= (u / (self.ring[0] + gap)) ** 2 + (v / (self.ring[1] + gap)) ** 2 > 1.0
        if self.block is not None:
            bu, bv, half_w, half_h = self.block
            ok &= ~((np.abs(u - bu) < half_w + gap) & (np.abs(v - bv) < half_h + gap))
        if self.middle:
            ok &= ((u / (0.60 * panel.panel_half * radius)) ** 2
                   + (v / (0.66 * panel.band_half * radius)) ** 2 > 1.0)
        return ok


def _edge_path(panel: Panel, top: bool, inset: float, drop: float, swag: float, ends: float,
               step: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """A line that hangs in swags along the top (or bottom) edge of a picture.

    inset: mm from the edge to the nearest the line comes to it - the points
    it hangs from along the top, the lowest point of each swag along the
    bottom; drop: how far each swag sags; swag: wanted length of one swag;
    ends: mm kept clear at either side.  Returns (dlam, phi, distance along
    the line in mm, the distances at which swags meet)."""
    radius = panel.radius
    if top:
        phi_hang = panel.band_half - inset / radius
    else:       # the swags hang down here too: it is their lowest points that are `inset` from the edge
        phi_hang = -(panel.band_half - (inset + drop) / radius)
    half = panel.panel_half - ends / (radius * math.cos(phi_hang))
    span = 2 * half * radius * math.cos(phi_hang)
    count = max(1, int(round(span / swag)))
    n = max(8, int(span / step))
    t = np.linspace(0.0, 1.0, n + 1)
    f = (t * count) % 1.0
    f[-1] = 1.0
    sag = drop * 4.0 * f * (1.0 - f)
    dlam = (2.0 * t - 1.0) * half
    phi = phi_hang - sag / radius
    seg = np.hypot(np.diff(dlam) * np.cos(phi[:-1]) * radius, np.diff(phi) * radius)
    along = np.concatenate([[0.0], np.cumsum(seg)])
    joints = [float(np.interp(k / count, t, along)) for k in range(count + 1)]
    return dlam, phi, along, joints


def _at(dlam, phi, along, where):
    """Sphere point(s) and direction of travel (radians from east) at the
    given distances along a path."""
    lon, lat = np.interp(where, along, dlam), np.interp(where, along, phi)
    ahead, behind = np.minimum(where + 0.3, along[-1]), np.maximum(where - 0.3, along[0])
    d_e = (np.interp(ahead, along, dlam) - np.interp(behind, along, dlam)) * np.cos(lat)
    d_n = np.interp(ahead, along, phi) - np.interp(behind, along, phi)
    return lon, lat, np.arctan2(d_n, d_e)


FIR_HALF = 4.6      # how far the needle tips of a bough reach from its middle line (deco units)
BAUBLE = 1.8        # radius of a bauble on a bough (deco units)


def soft_zone(phi):
    """How much bolder the needles of a bough are drawn at latitude phi
    (radians): 1.0 up to a little above the middle of the ornament, a quarter
    more from 40 degrees up.

    Below the middle the printer can build any edge on the outside of the
    wall and detail comes out crisp.  Higher up more and more of every edge
    has to be made on the inside (imaging.outside_first), where the light
    spreads in the plastic and blurs it by about a millimetre, so fine
    strokes close together fade there.  A little bolder keeps them showing;
    much bolder (1.8 was tried) and they stop reading as needles and look
    like the leaves of a laurel wreath."""
    t = np.clip((np.degrees(phi) - 12.0) / 28.0, 0.0, 1.0)
    return 1.0 + 0.25 * t * t * (3.0 - 2.0 * t)


def _fir(sheet: Sheet, dlam, phi, along, s: float, closed: bool = False, seed: int = 11,
         half=None) -> None:
    """An evergreen bough along a path: a dark branch thick with glowing
    needles, all of them leaning the same way along it.

    Many thin needles set close together are what make it fir.  Fewer, fatter
    leaves a little apart - which is how this was drawn at first - read as a
    laurel branch.  Each needle is still wider than two lines of a 0.4 mm
    nozzle, and the dark between them is solid, so the bough prints as a band
    with a ragged edge and the needles as grooves in it.  (For a closed path,
    pass it with its first point repeated at the end.)

    half: how far the needle tips reach from the path, in mm - one number, or
    one per point of the path (default: FIR_HALF deco units)."""
    rng = np.random.default_rng(seed)
    total = float(along[-1])
    reach = np.broadcast_to(np.asarray(FIR_HALF * s if half is None else half, dtype=np.float64),
                            np.shape(along))
    # Needles are set closer together low down, where they print crisp, than high up, where
    # only bolder ones show at all (see soft_zone): count them off along the path accordingly.
    bold = soft_zone(np.asarray(phi, dtype=np.float64))
    steps = np.diff(along) / (2.15 * s * 0.5 * (bold[:-1] + bold[1:]))
    tally = np.concatenate([[0.0], np.cumsum(steps)])
    count = max(4, int(round(tally[-1])))
    k = np.arange(count)
    # the two sides are out of step, like needles on a twig
    turn = np.concatenate([(k + 0.5), (k + 1.0)]) * tally[-1] / count
    where = np.interp(turn % tally[-1] if closed else np.minimum(turn, tally[-1]), tally, along)
    side = np.concatenate([np.ones(count), -np.ones(count)])
    lon, lat, heading = _at(dlam, phi, along, where)
    long_ones = (np.arange(2 * count) % 2 == 0)         # long and short turn and turn about
    room = np.interp(where, along, reach)
    if float(np.ptp(reach)) > 1e-9 and not closed:
        # where the room changes along the path, a needle gets the least there is anywhere
        # it lies: it leans back along the path, towards what may be a narrower part
        for step in (-1.0, -0.5, 0.5, 1.0):
            room = np.minimum(room, np.interp(np.clip(where + step * room, 0.0, total), along, reach))
    length = room * np.where(long_ones, rng.uniform(1.12, 1.26, 2 * count),
                             rng.uniform(0.78, 0.94, 2 * count))
    lean = np.radians(128.0) + rng.uniform(-0.10, 0.10, 2 * count)     # from the way the path runs
    line = (dlam[:-1], phi[:-1]) if closed else (dlam, phi)
    body = 1.56 * (reach[:-1] if closed else reach)
    # the body of the bough: a dark mass the short needles lie on and the long ones reach out of
    sheet.ribbon(*line, body, "dark", closed=closed)
    sheet.needles(lon, lat, heading + side * lean, length, 1.12 * s * soft_zone(lat), "light",
                  edge=EDGE_MM)
    sheet.ribbon(*line, np.minimum(1.1 * s, 0.3 * body), "dark", closed=closed)


def _baubles(sheet: Sheet, dlam, phi, along, s: float, spans: list[tuple[float, float]],
             every: float = 11.0) -> None:
    """Glowing baubles on a bough, evenly spread over each of the given
    stretches of its path (none at their ends, where the bows go)."""
    for a, b in spans:
        count = max(1, int(round((b - a) / (every * s))) - 1)
        for k in range(count):
            lon, lat, _heading = _at(dlam, phi, along, a + (k + 1) * (b - a) / (count + 1))
            sheet.disc((float(lon), float(lat)), BAUBLE * s, "light", edge=0.8 * s)


def _bow(sheet: Sheet, at: tuple[float, float], s: float, splay: float = 0.0) -> None:
    """A ribbon bow: two loops, two tails and a knot.  splay swings the tails
    out to the sides by that many radians (on a wreath they lie along it
    instead of hanging down out of the picture)."""
    t = np.linspace(0.0, 1.0, 15)
    parts = []
    cs, sn = math.cos(splay), math.sin(splay)
    for side in (-1.0, 1.0):
        # loop: a fat teardrop from the knot out to the side
        reach = side * (0.2 + 5.0 * t) * s
        parts.append((np.concatenate([reach, reach[::-1]]),
                      np.concatenate([2.5 * s * np.sin(math.pi * t) ** 0.7 + 0.7 * s * t,
                                      (-2.1 * s * np.sin(math.pi * t) ** 0.7 + 0.7 * s * t)[::-1]])))
        # tail: a tapering strip hanging down and out, with a notched end
        east = np.array([0.1, 2.0, 4.9, 3.6, 2.2]) * s
        north = np.array([-0.2, -0.5, -6.0, -5.0, -6.4]) * s
        parts.append((side * (east * cs - north * sn), east * sn + north * cs))
    with sheet.group("light", EDGE_MM):
        for east, north in parts:
            sheet.shape(at, east, north, "light")
    for side in (-1.0, 1.0):            # the fold in each loop
        fold = np.linspace(1.5, 3.9, 7) * s
        lon, lat = sheet.offset(at[0], at[1], side * fold, 0.22 * fold)
        sheet.ribbon(lon, lat, 0.75 * s, "dark")
    sheet.disc(at, 1.05 * s, "light", edge=0.75 * s)


def _garland(sheet: Sheet, top: bool, s: float) -> None:
    """A fir garland with baubles and bows, hung in swags along one edge."""
    dlam, phi, along, joints = _edge_path(sheet.panel, top, inset=5.6 * s, drop=4.6 * s,
                                          swag=24.0 * s, ends=5.8 * s)
    _fir(sheet, dlam, phi, along, s)
    _baubles(sheet, dlam, phi, along, s, list(zip(joints[:-1], joints[1:])), every=8.5)
    for where in joints:
        lon, lat, _heading = _at(dlam, phi, along, where)
        _bow(sheet, (float(lon), float(lat)), 0.9 * s)


def _light_wire(panel: Panel, s: float):
    return _edge_path(panel, True, inset=2.6 * s, drop=3.4 * s, swag=24.0 * s, ends=1.5 * s)


LIGHT_TILT = 0.12           # radians a bulb hangs askew, one way and the other in turn
LIGHT_BODY = (1.8, 4.6, 1.75)   # a bulb: mm from the wire to its body, its length, half its width


def light_spots(panel: Panel, s: float) -> list[tuple[float, float]]:
    """Where the bulbs of a string of fairy lights hang from its wire: spread
    evenly over each swag, none at the points where two swags meet, and never
    closer together than leaves a clear millimetre of dark between two bulbs
    that lean towards each other."""
    dlam, phi, along, joints = _light_wire(panel, s)
    spots = []
    for a, b in zip(joints[:-1], joints[1:]):
        count = max(1, int((b - a) / (7.0 * s)))
        for k in range(count):
            lon, lat, _heading = _at(dlam, phi, along, a + (k + 0.5) * (b - a) / count)
            spots.append((float(lon), float(lat)))
    return spots


def _lights(sheet: Sheet, s: float) -> None:
    """A string of fairy lights along the top edge."""
    dlam, phi, along, _joints = _light_wire(sheet.panel, s)
    sheet.ribbon(dlam, phi, 0.9 * s + 2 * EDGE_MM, "light")
    sheet.ribbon(dlam, phi, 0.9 * s, "dark")
    t = np.linspace(0.0, 1.0, 13)
    for k, (lon, lat) in enumerate(light_spots(sheet.panel, s)):
        hang = -0.5 * math.pi + (LIGHT_TILT if k % 2 else -LIGHT_TILT)     # down, a little askew
        ca, sa = math.cos(hang), math.sin(hang)

        def put(a, b, tone, edge, lon=lon, lat=lat, ca=ca, sa=sa):
            a, b = np.asarray(a), np.asarray(b)
            sheet.shape((float(lon), float(lat)), a * ca - b * sa, a * sa + b * ca, tone, edge)

        put(np.array([0.2, 1.9, 1.9, 0.2]) * s, np.array([0.9, 0.9, -0.9, -0.9]) * s, "dark", 0.0)
        body = LIGHT_BODY[0] * s + LIGHT_BODY[1] * s * t
        bulge = LIGHT_BODY[2] * s * np.sin(math.pi * t ** 0.8) ** 0.75
        put(np.concatenate([body, body[::-1]]), np.concatenate([bulge, -bulge[::-1]]), "light", EDGE_MM)


def _icicles(sheet: Sheet, s: float) -> None:
    """Snow along the top edge with icicles hanging from it."""
    rng = np.random.default_rng(23)
    panel, radius = sheet.panel, sheet.panel.radius
    span = 2 * panel.panel_half * radius * math.cos(panel.band_half)
    count = max(3, int(round(span / (5.6 * s))))
    east = [0.0]
    down = [2.6 * s]
    x = 0.0
    for k in range(count):
        pitch = span / count
        base = pitch * rng.uniform(0.5, 0.72)           # wide enough to keep a light core
        length = s * (rng.uniform(3.5, 5.5) if k % 2 else rng.uniform(6.5, 10.0))
        left = x + (pitch - base) / 2
        east += [left, left + base * 0.5, left + base, x + pitch]
        down += [2.6 * s + rng.uniform(-0.5, 0.5) * s, 2.6 * s + length,
                 2.6 * s + rng.uniform(-0.5, 0.5) * s, 2.6 * s + rng.uniform(-0.3, 0.6) * s]
        x += pitch
    east, down = np.asarray(east), np.asarray(down)
    # round the tips a little: three points per icicle is too sharp to print
    fine = np.linspace(0.0, span, int(span / 0.3) + 2)
    edge = ndimage.gaussian_filter1d(np.interp(fine, east, down), 1.2, mode="nearest")
    phi = panel.band_half - edge / radius
    # from one side of the picture to the other, and a little over (only the picture is kept)
    dlam = (fine / span * 2.0 - 1.0) * (panel.panel_half + 2.5 / radius)
    top = np.full_like(dlam, panel.band_half + 2.0 / radius)
    sheet.ribbon(dlam, phi, 2 * EDGE_MM, "dark")
    sheet.fill(np.concatenate([dlam, dlam[::-1]]), np.concatenate([phi, top]), "light")


def _holly_leaf(length: float, width: float, bend: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Outline of a holly leaf lying along +east from its stem (mm)."""
    t = np.linspace(0.0, 1.0, 61)
    body = np.sin(math.pi * t) ** (0.6 + 0.6 * t)           # rounded at the stem, pointed at the tip
    spikes = 1.0 - 0.42 * np.abs(np.sin(3 * math.pi * t)) ** 0.85
    half = 0.5 * width * body * spikes
    bow = bend * length * np.sin(math.pi * t)
    return (np.concatenate([t * length, (t * length)[::-1]]),
            np.concatenate([bow + half, (bow - half)[::-1]]))


_HOLLY_REACH = (21.5, 22.0)     # how far a sprig reaches from its edge and from its side


def _holly(sheet: Sheet, top: bool, s: float) -> None:
    """A sprig of holly in each corner of one edge."""
    panel, radius = sheet.panel, sheet.panel.radius
    sign = 1.0 if top else -1.0
    phi0 = sign * (panel.band_half - 7.0 * s / radius)
    for side in (-1.0, 1.0):
        at = (side * (panel.panel_half - 7.5 * s / (radius * math.cos(phi0))), phi0)
        # three leaves fanning out from the corner towards the middle of the picture
        for k, turn in enumerate(math.radians(a) for a in (8.0, -38.0, -84.0)):
            length, width = (12.5 if k == 1 else 10.5) * s, (5.6 if k == 1 else 5.0) * s
            east, north = _holly_leaf(length, width, bend=0.05 if k != 1 else 0.0)
            east = east + 1.2 * s
            ca, sa = math.cos(turn), math.sin(turn)
            sheet.shape(at, (east * ca - north * sa) * -side, (east * sa + north * ca) * sign,
                        "light", EDGE_MM)
            if top:         # up here a rib would only blur the leaf it is drawn on (see soft_zone)
                continue
            rib = np.linspace(0.14, 0.86, 9) * length + 1.2 * s
            lon, lat = sheet.offset(at[0], at[1], rib * ca * -side, rib * sa * sign)
            sheet.ribbon(lon, lat, 0.75 * s, "dark")
        berry = 1.8 if top else 1.45
        for de, dn in ((-1.6, 1.2), (1.4, 1.5), (0.0, -1.3)):
            lon, lat = sheet.offset(at[0], at[1], de * berry / 1.45 * s * -side, dn * berry / 1.45 * s * sign)
            sheet.disc((float(lon), float(lat)), berry * s, "light", edge=0.75 * s)


def _snow(sheet: Sheet, s: float, trees: bool = False) -> None:
    """A bank of snow along the bottom edge, with fir trees standing in it if
    asked."""
    rng = np.random.default_rng(31)
    panel, radius = sheet.panel, sheet.panel.radius
    span = 2 * panel.panel_half * radius * math.cos(panel.band_half)
    x = np.linspace(0.0, span, int(span / 0.4) + 2)
    height = np.full_like(x, 4.2 * s)
    for _ in range(max(3, int(span / (16.0 * s)))):
        centre, wide = rng.uniform(0.0, span), rng.uniform(7.0, 13.0) * s
        height += rng.uniform(1.2, 3.4) * s * np.exp(-0.5 * ((x - centre) / wide) ** 2)
    t = x / span * 2.0 - 1.0
    phi = -panel.band_half + height / radius
    # from one side of the picture to the other, and a little over (only the picture is kept)
    dlam = t * (panel.panel_half + 2.5 / radius)
    if trees:
        count = max(2, int(round(span / (15.0 * s))))
        for k in range(count):
            # taller towards the sides, where they frame the picture instead of covering it
            where = (k + rng.uniform(0.25, 0.75)) / count * 2.0 - 1.0
            tall = s * (8.0 + 7.0 * abs(where) ** 1.5 + rng.uniform(-1.0, 1.5))
            wide = tall * rng.uniform(0.52, 0.62)
            foot = float(np.interp(where, t, height)) - 1.4 * s
            at = (float(np.interp(where, t, dlam)), -panel.band_half + foot / radius)
            east, north = [0.0], [tall]
            tiers = 3
            for tier in range(tiers):
                f = (tier + 1) / tiers
                east += [0.5 * wide * (0.36 + 0.64 * f)]
                north += [tall * (1.0 - (0.2 + 0.8 * f))]
                if tier < tiers - 1:
                    east += [0.5 * wide * (0.36 + 0.64 * f) * 0.46]
                    north += [tall * (1.0 - (0.2 + 0.8 * f)) + 0.05 * tall]
            east = np.array(east + [-e for e in east[:0:-1]])
            north = np.array(north + north[:0:-1])
            sheet.shape(at, east, north, "dark", EDGE_MM)
    below = np.full_like(dlam, -panel.band_half - 2.0 / radius)
    sheet.ribbon(dlam, phi, 2 * EDGE_MM, "dark")
    sheet.fill(np.concatenate([dlam, dlam[::-1]]), np.concatenate([phi, below]), "light")


def _inset_path(panel: Panel, inset: float, corner: float, step: float = 0.4):
    """A closed line `inset` mm inside the edge of the picture, with rounded
    corners.  Returns (dlam, phi, distance along it in mm)."""
    radius = panel.radius
    narrow = panel.panel_half * radius * math.cos(panel.band_half)      # half the width, top and bottom
    corner = max(0.5, min(corner, panel.band_half * radius - inset - 0.5, narrow - inset - 0.5))
    lon, lat = [], []
    # walk it anticlockwise from the bottom right-hand corner
    for sx, sy, start in ((1, -1, -90.0), (1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0)):
        phi_c = sy * (panel.band_half - (inset + corner) / radius)
        lam_c = sx * (panel.panel_half - (inset + corner) / (radius * math.cos(phi_c)))
        turn = np.radians(start + np.linspace(0.0, 90.0, 9))
        a, b = offset(lam_c, phi_c, corner * np.cos(turn), corner * np.sin(turn), radius)
        lon.append(a)
        lat.append(b)
    lon, lat = np.concatenate(lon), np.concatenate(lat)
    # between the corners: fill in points, so that each side follows its edge of the picture
    lon2, lat2 = [], []
    for k in range(len(lon)):
        a, b = (lon[k], lat[k]), (lon[(k + 1) % len(lon)], lat[(k + 1) % len(lon)])
        gap = math.hypot((b[0] - a[0]) * math.cos(a[1]) * radius, (b[1] - a[1]) * radius)
        n = max(1, int(gap / step))
        f = np.arange(n) / n
        lat_k = a[1] + f * (b[1] - a[1])
        if (k + 1) % 9 == 0 and a[0] * b[0] > 0:            # down a side: stay `inset` from it
            side = math.copysign(1.0, a[0])
            lon2.append(side * (panel.panel_half - inset / (radius * np.cos(lat_k))))
        else:
            lon2.append(a[0] + f * (b[0] - a[0]))
        lat2.append(lat_k)
    lon, lat = np.concatenate(lon2), np.concatenate(lat2)
    seg = np.hypot((np.roll(lon, -1) - lon) * np.cos(lat) * radius, (np.roll(lat, -1) - lat) * radius)
    return lon, lat, np.concatenate([[0.0], np.cumsum(seg)])


def _ground(sheet: Sheet, lon, lat) -> None:
    """What a frame stands on: everything outside the closed line becomes
    plain, like the rest of the ball, and inside it the picture stays.  So no
    sliver of the photo is left between a frame and the edge of its picture."""
    wide = max(sheet.size) * 2
    sheet.draw.rectangle([-wide, -wide, wide, wide], fill=_FILL["plain"])
    sheet.fill(lon, lat, "clear")


LINE_FRAME = 1.2            # width of the line of a line frame (deco units)
CANDY_BAND = 5.0            # width of a candy-cane border (deco units)
WREATH_BOUGH = FIR_HALF + 0.7   # from the middle line of a wreath to its outermost edge (deco units)


def _line_frame(sheet: Sheet, s: float) -> None:
    """A fine frame: a glowing line with a dark edge, right at the edge of the picture."""
    lon, lat, _along = _inset_path(sheet.panel, LINE_FRAME * s / 2 + EDGE_MM, 4.5 * s)
    _ground(sheet, lon, lat)
    sheet.ribbon(lon, lat, LINE_FRAME * s + 2 * EDGE_MM, "dark", closed=True)
    sheet.ribbon(lon, lat, LINE_FRAME * s, "light", closed=True)


def _candy_frame(sheet: Sheet, s: float) -> None:
    """A candy-cane border: a broad glowing band with bold dark stripes on the
    slant, like the red on a stick of candy.  It runs right along the edge of
    the picture."""
    band = CANDY_BAND * s
    lon, lat, along = _inset_path(sheet.panel, band / 2 + EDGE_MM, 5.2 * s)
    _ground(sheet, lon, lat)
    total = float(along[-1])
    sheet.ribbon(lon, lat, band + 2 * EDGE_MM, "dark", closed=True)
    sheet.ribbon(lon, lat, band, "light", closed=True)
    # stripes at 45 degrees, one broad and one narrow turn and turn about; each follows the
    # band round the corners instead of cutting across them
    count = max(4, int(round(total / (7.6 * s))))
    pitch = total / count
    half = band / 2 + 0.3
    path = (np.append(lon, lon[0]), np.append(lat, lat[0]))
    across = np.linspace(-half, half, 9)
    for shift, wide in ((0.0, 2.5 * s), (0.5 * pitch + 0.1 * s, 1.15 * s)):
        for k in range(count):
            centre = (k + 0.25) * pitch + shift
            edge_lon, edge_lat = [], []
            for where, b in ((centre + across - wide / 2, across),
                             (centre + across[::-1] + wide / 2, across[::-1])):
                p_lon, p_lat, heading = _at(*path, along, where % total)
                e_lon, e_lat = sheet.offset(p_lon, p_lat, -np.sin(heading) * b, np.cos(heading) * b)
                edge_lon.append(e_lon)
                edge_lat.append(e_lat)
            sheet.fill(np.concatenate(edge_lon), np.concatenate(edge_lat), "dark")


def wreath_axes(panel: Panel, s: float) -> tuple[float, float, float]:
    """(a, b, bough): the half-axes in mm of the oval the middle line of a
    round wreath follows - the widest that stays inside the picture - and how
    far its needles reach either side of that line."""
    bough = WREATH_BOUGH * s
    half_width = _half_width_at(panel)
    b = max(4.0, panel.band_half * panel.radius - bough - 0.6)
    turn = np.linspace(0.0, 0.5 * math.pi, 16)
    a = b * 1.12
    for _ in range(80):
        if all(a * math.cos(t) + bough + 0.6 <= half_width(b * math.sin(t)) for t in turn):
            break
        a *= 0.985
    return a, b, bough


def _wreath_trim(sheet: Sheet, lon, lat, along, s: float, foot: float) -> None:
    """Baubles all round a wreath and a bow at its foot (`foot`: how far along
    the line the lowest point is)."""
    total = float(along[-1])
    path = (np.append(lon, lon[0]), np.append(lat, lat[0]))
    clear = 7.5 * s                                 # the bow's half-width
    count = max(4, int(round((total - 2 * clear) / (11.5 * s))))
    for k in range(count):
        where = (foot + clear + (k + 0.5) * (total - 2 * clear) / count) % total
        at_lon, at_lat, _heading = _at(*path, along, where)
        sheet.disc((float(at_lon), float(at_lat)), BAUBLE * s, "light", edge=0.8 * s)
    at_lon, at_lat, _heading = _at(*path, along, foot % total)
    _bow(sheet, (float(at_lon), float(at_lat)), 1.25 * s, splay=math.radians(34.0))


def _closed_line(panel: Panel, lon, lat):
    """Distance along a closed line of sphere points, in mm (one longer than
    the line: the last value is its whole length)."""
    seg = np.hypot((np.roll(lon, -1) - lon) * np.cos(lat) * panel.radius,
                   (np.roll(lat, -1) - lat) * panel.radius)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _wreath(sheet: Sheet, s: float) -> None:
    """A round Christmas wreath of fir with baubles and a bow.  The picture
    shows inside it; the rest of the panel is plain."""
    a, b, _bough = wreath_axes(sheet.panel, s)
    turn = np.linspace(0.0, 2 * math.pi, 361)[:-1]
    lon, lat = sheet.offset(0.0, 0.0, a * np.cos(turn), b * np.sin(turn))
    _ground(sheet, lon, lat)
    along = _closed_line(sheet.panel, lon, lat)
    _fir(sheet, np.append(lon, lon[0]), np.append(lat, lat[0]), along, s, closed=True)
    _wreath_trim(sheet, lon, lat, along, s, foot=0.75 * float(along[-1]))


def _wreath_square(sheet: Sheet, s: float) -> None:
    """The same wreath laid along the edges of the picture, so that all of
    the picture but its border shows."""
    lon, lat, along = _inset_path(sheet.panel, WREATH_BOUGH * s + 0.3, 9.0 * s)
    _ground(sheet, lon, lat)
    _fir(sheet, np.append(lon, lon[0]), np.append(lat, lat[0]), along, s, closed=True)
    low = np.flatnonzero(lat < 0.0)
    foot = float(along[low[np.argmin(np.abs(lon[low]))]])       # the middle of the bottom side
    _wreath_trim(sheet, lon, lat, along, s, foot=foot)


def _flake(size: float, grow: float) -> tuple[np.ndarray, np.ndarray]:
    """The strokes of a snowflake as pointed bars: (bars, 5 corners) east and
    north of its middle, in mm.  grow widens and lengthens every bar."""
    arm = max(0.85, 0.24 * size)
    bars = []                           # start east, start north, angle, length, width
    for k in range(6):
        a = k * math.pi / 3
        bars.append((0.0, 0.0, a, size, arm))
        if size >= 3.2:                 # big enough for a pair of side branches
            for side in (-1.0, 1.0):
                bars.append((0.58 * size * math.cos(a), 0.58 * size * math.sin(a),
                             a + side * math.pi / 3, 0.36 * size, 0.9 * arm))
    e0, n0, angle, length, width = (np.array(col)[:, None] for col in zip(*bars))
    g = grow / 2
    half = width / 2 + g
    along = np.concatenate([np.full_like(length, -g), length - width / 2, length + g,
                            length - width / 2, np.full_like(length, -g)], axis=1)
    across = half * np.array([[1.0, 1.0, 0.0, -1.0, -1.0]])
    ca, sa = np.cos(angle), np.sin(angle)
    return e0 + along * ca - across * sa, n0 + along * sa + across * ca


def _snowflake(sheet: Sheet, at: tuple[float, float], size: float, turn: float,
               edge: float = EDGE_MM) -> None:
    ca, sa = math.cos(turn), math.sin(turn)
    hub = 0.75 * max(0.85, 0.24 * size)
    for tone, grow in (("dark", 2 * edge), ("light", 0.0)):
        if tone == "dark" and edge <= 0.0:
            continue
        east, north = _flake(size, grow)
        sheet.shapes(at[0], at[1], east * ca - north * sa, east * sa + north * ca, tone)
        sheet.disc(at, hub + grow / 2, tone)


def _star(sheet: Sheet, at: tuple[float, float], size: float, turn: float,
          edge: float = EDGE_MM) -> None:
    k = np.arange(10)
    r = np.where(k % 2 == 0, size, 0.46 * size)
    a = turn + math.pi / 2 + k * math.pi / 5
    sheet.shape(at, r * np.cos(a), r * np.sin(a), "light", edge)


def _heart(sheet: Sheet, at: tuple[float, float], size: float, turn: float) -> None:
    t = np.linspace(0.0, 2 * math.pi, 40, endpoint=False)
    east = 16 * np.sin(t) ** 3
    north = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t) + 2.5
    east, north = east * size / 16.0, north * size / 16.0
    ca, sa = math.cos(turn), math.sin(turn)
    sheet.shape(at, east * ca - north * sa, east * sa + north * ca, "light", EDGE_MM)


# kind -> (smallest, largest size in mm, how many compared with snowflakes)
_SCATTER = {"snowflakes": (2.5, 4.4, 1.0), "snow": (0.8, 1.5, 1.8), "stars": (2.3, 3.9, 1.0),
            "hearts": (2.1, 3.4, 1.0)}


def claims_for(slot: ImageSlot, panel: Panel, text: Lettering | None) -> _Claims:
    """The parts of a slot's picture that its lettering and decorations take up."""
    s = deco_unit(slot, panel)
    top, bottom, sides = reserved(slot, panel)
    claims = _Claims(top=_REACH_TOP.get(slot.deco_top, 0.0) * s,
                     bottom=_REACH_BOTTOM.get(slot.deco_bottom, 0.0) * s, sides=sides,
                     middle=bool(slot.path))
    if slot.deco_bottom == "trees":         # the trees stand taller than the strip the
        claims.bottom = 18.0 * s            # lettering keeps out of
    for is_top, kind in ((True, slot.deco_top), (False, slot.deco_bottom)):
        if kind == "holly":
            claims.corners.append((is_top, _HOLLY_REACH[0] * s, _HOLLY_REACH[1] * s))
    if slot.deco_frame == "wreath":
        a, b, bough = wreath_axes(panel, frame_unit(slot, panel))
        claims.ring = (a + bough, b + bough)
        claims.middle = False               # the photo is inside the wreath: sprinkle round it
    if text is not None:
        claims.block = (text.u, text.v, text.width * text.stretch / 2, text.height / 2)
    return claims


def scatter_points(slot: ImageSlot, panel: Panel) -> list[tuple[float, float, float, float]]:
    """Where the scattered shapes of a slot's picture go: (dlam, phi, size in
    mm, turn 0..1).

    Worked out on the sphere alone - never from the drawn sheet - so that the
    quick preview, the sharp one and the exported model all put every shape in
    exactly the same place."""
    kind = slot.deco_scatter
    if kind not in _SCATTER:
        return []
    claims = claims_for(slot, panel, lettering_for(slot, panel))
    s, amount = deco_unit(slot, panel), float(slot.deco_amount)
    radius = panel.radius
    rng = np.random.default_rng(1000 + int(slot.deco_seed))
    lo, hi, many = _SCATTER[kind]
    lo, hi = lo * s, hi * s
    area = 4 * panel.panel_half * math.sin(panel.band_half) * radius ** 2
    wanted = int(round((5 + 50 * amount) * area / 4600.0 * many))
    if wanted <= 0:
        return []
    spread = 0.5 * math.sqrt(area / wanted)             # evenly, not in clumps
    tries = wanted * 80
    # darts thrown evenly over the picture; each shape must clear everything by its own
    # size, so small ones find room where a large one would not
    phi = np.arcsin(rng.uniform(-1.0, 1.0, tries) * math.sin(panel.band_half))
    dlam = rng.uniform(-1.0, 1.0, tries) * panel.panel_half
    sizes = rng.uniform(lo, hi, tries)
    turns = rng.uniform(0.0, 1.0, tries)
    free = claims.free(panel, dlam, phi, sizes + 1.0)
    points = np.stack([np.cos(phi) * np.cos(dlam), np.cos(phi) * np.sin(dlam), np.sin(phi)], axis=1)
    chosen: list[int] = []
    for i in np.flatnonzero(free):
        if len(chosen) >= wanted:
            break
        if chosen:
            gap = radius * np.sqrt(((points[chosen] - points[i]) ** 2).sum(axis=1))
            if (gap < np.maximum(spread, sizes[chosen] + sizes[i] + 1.2)).any():
                continue
        chosen.append(int(i))
    return [(float(dlam[i]), float(phi[i]), float(sizes[i]), float(turns[i])) for i in chosen]


def _scatter(sheet: Sheet, slot: ImageSlot) -> None:
    kind = slot.deco_scatter
    for dlam, phi, size, turn in scatter_points(slot, sheet.panel):
        if kind == "snowflakes":
            _snowflake(sheet, (dlam, phi), size, turn * math.pi / 3)
        elif kind == "stars":
            _star(sheet, (dlam, phi), size, (turn - 0.5) * 0.6)
        elif kind == "hearts":
            _heart(sheet, (dlam, phi), size, (turn - 0.5) * 0.7)
        else:
            sheet.disc((dlam, phi), size, "light", edge=EDGE_MM)


# ---------------------------------------------------------------------------
# the finished layer
# ---------------------------------------------------------------------------
@dataclass
class Layer:
    """What lies over one picture, ready to be read at any point of it."""

    panel: Panel
    projection: str
    u_max: float
    v_max: float
    k: float                    # sheet pixels per radian
    hidden: np.ndarray          # uint8, 0..255: how much of the picture underneath is hidden
    light: np.ndarray           # ... and how much of the point is light
    plain: np.ndarray           # ... or plain frame (the rest of what hides it is dark)
    lettering: Lettering | None = None

    @staticmethod
    def of(sheet: Sheet, lettering: Lettering | None = None, soften: bool = True) -> "Layer":
        """The layer a finished sheet makes.  It is softened by about one sample
        of the model first, so that reading it back does not alias."""
        image = sheet.image.filter(ImageFilter.GaussianBlur(0.42 * SUPERSAMPLE)) if soften \
            else sheet.image
        hidden, light, plain = (np.array(channel, dtype=np.uint8) for channel in image.split())
        return Layer(panel=sheet.panel, projection=sheet.projection, u_max=sheet.u_max,
                     v_max=sheet.v_max, k=sheet.k, hidden=hidden, light=light, plain=plain,
                     lettering=lettering)

    def sample(self, dlam: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(hidden, light, plain), each 0..1, at the given points of the picture."""
        u, v = flat_uv(dlam, phi, self.projection)
        where = [(self.v_max - v) * self.k + 1.0, (u + self.u_max) * self.k + 1.0]
        return tuple(ndimage.map_coordinates(channel, where, order=1, output=np.float32,
                                             mode="constant", cval=0.0) * np.float32(1.0 / 255.0)
                     for channel in (self.hidden, self.light, self.plain))


_SLOT_FIELDS = ("text", "text_font", "text_tone", "text_size", "text_x", "text_y", "text_outline",
                "text_bold", "deco_top", "deco_bottom", "deco_frame", "deco_scatter", "deco_amount",
                "deco_size", "deco_seed", "backdrop")
assert set(_SLOT_FIELDS) <= {f.name for f in fields(ImageSlot)}


def has_overlay(slot: ImageSlot) -> bool:
    """True if anything is drawn over this slot's picture."""
    return bool(slot.decorated)


# The most of a picture a border may take: its width on each side, as a share of the
# smaller of the picture's half-height and its half-width where it is narrowest.
_FRAME_SHARE = {"wreath": 0.28, "wreath_square": 0.28, "candy": 0.24}


def frame_unit(slot: ImageSlot, panel: Panel) -> float:
    """The scale a picture's border is drawn at: deco_unit(), but never so
    large that the border swallows the picture.  On the narrow pictures of a
    ball with four, or on low ones, a wreath at full size would leave a slit
    of photo, fold over itself in the corners and push its bow off the edge."""
    s = deco_unit(slot, panel)
    share = _FRAME_SHARE.get(slot.deco_frame)
    if share is None:
        return s
    tall = panel.band_half * panel.radius
    least = min(tall, _half_width_at(panel)(tall))
    across = CANDY_BAND if slot.deco_frame == "candy" else WREATH_BOUGH
    return min(s, share * least / across)


def deco_unit(slot: ImageSlot, panel: Panel) -> float:
    """The scale decorations are drawn at: 1.0 is their natural size on a
    picture 70 mm tall (an 80 mm ornament).  They grow and shrink with the
    picture, but only so far - a sprig half the size would be too fine to
    print - and the slot's own Size setting comes on top."""
    tall = 2 * panel.band_half * panel.radius
    return float(slot.deco_size) * min(1.6, max(0.75, tall / 70.0))


_layers: dict[tuple, Layer] = {}
_layers_lock = threading.Lock()         # the preview and an export may ask at the same time


def layer_for(slot: ImageSlot, panel: Panel, pitch_mm: float) -> Layer | None:
    """The lettering and decorations of a slot for a picture of this size,
    drawn finely enough for a model sampled every pitch_mm (cached)."""
    if not has_overlay(slot):
        return None
    key = (tuple(getattr(slot, name) for name in _SLOT_FIELDS), bool(slot.path),
           round(panel.radius, 4), round(panel.panel_half, 6), round(panel.band_half, 6),
           round(pitch_mm, 4))
    with _layers_lock:
        hit = _layers.get(key)
    if hit is None:
        hit = _render(slot, panel, pitch_mm)
        with _layers_lock:
            while len(_layers) >= 16:
                _layers.pop(next(iter(_layers)))
            _layers[key] = hit
    return hit


def _render(slot: ImageSlot, panel: Panel, pitch_mm: float) -> Layer:
    sheet = Sheet(panel, SUPERSAMPLE / max(pitch_mm, 0.05))
    s = deco_unit(slot, panel)
    text = lettering_for(slot, panel)
    # on a bright panel light shapes would not show: there the decorations are drawn dark
    sheet.swap = not slot.path and slot.backdrop == "light"
    border = frame_unit(slot, panel)
    if slot.deco_frame == "wreath":
        _wreath(sheet, border)
    elif slot.deco_frame == "wreath_square":
        _wreath_square(sheet, border)
    elif slot.deco_frame == "line":
        _line_frame(sheet, border)
    elif slot.deco_frame == "candy":
        _candy_frame(sheet, border)
    if slot.deco_bottom == "garland":
        _garland(sheet, False, s)
    elif slot.deco_bottom in ("snow", "trees"):
        _snow(sheet, s, trees=slot.deco_bottom == "trees")
    elif slot.deco_bottom == "holly":
        _holly(sheet, False, s)
    if slot.deco_top == "garland":
        _garland(sheet, True, s)
    elif slot.deco_top == "lights":
        _lights(sheet, s)
    elif slot.deco_top == "icicles":
        _icicles(sheet, s)
    elif slot.deco_top == "holly":
        _holly(sheet, True, s)
    _scatter(sheet, slot)
    sheet.swap = False
    if text is not None:
        _draw_lettering(sheet, slot, text)
    return Layer.of(sheet, text)


def apply(thickness: np.ndarray, layer: Layer, dlam: np.ndarray, phi: np.ndarray,
          p: Params) -> np.ndarray:
    """Wall thickness with a layer laid over it: light is the thinnest wall,
    dark the thickest and plain the frame."""
    hidden, light, plain = layer.sample(dlam, phi)
    dark = np.clip(hidden - light - plain, 0.0, 1.0)
    return thickness * (1.0 - hidden) + p.t_min * light + p.t_frame * plain + p.t_max * dark


# ---------------------------------------------------------------------------
# the plain parts of the ball
# ---------------------------------------------------------------------------
# Between the pictures, above them and below them the ball is plain frame.
# Those parts can carry decorations of their own, drawn with the same tools as
# the ones on a picture - glowing shapes - and built by the same rules: below
# the middle of the ball they are cut into the outside of the wall and come
# out crisp; higher up they have to be made on the inside, where they show
# softly when the ornament is lit and not at all unlit.
#
# Two things stay plain on purpose.  The divider at the back carries the seam
# (every layer starts there), and a ring of decoration leaves a gap where it
# crosses that line: a seam is best hidden in thick plain plastic.  And the
# closed end, underneath, is only ever built UP: its wall is the roof over
# the hollow of the ball and is not to be thinned.
BALL_EDGE = 0.9             # mm kept clear between a decoration and the edge of the part it is on
# A shape on a picture has a dark edge round it, to read against any photo.  On the plain
# ball it needs none - it glows against plain frame - and is better without: every raised
# edge on the lower half of the ball ends in a ridge that each layer cuts off as a speck of
# its own.  (Sliced: stars between the pictures and snowflakes round the bottom made 563
# such specks in the bottom ring with edges and 278 without, and the two wall loops that
# had started at one of them started at the back again.)
BALL_RIM = 0.0
RING_BACK_GAP = 2.2         # mm kept clear either side of the seam where a ring crosses it
POLE_RAISE = 0.6            # mm a shape on the closed end stands proud of the ball
POLE_EDGE = 1.6             # mm kept clear between that shape and where the wall changes shape
MOTIF_MAX = 3.3             # largest star, snowflake or bead ring: mm from its middle to its tips
MOTIF_MIN = 1.5             # ... and the smallest worth printing
FRONT_LON = BACK_LON + math.pi
# The least room each decoration needs, in mm from the middle line of the part it is on to
# the nearest it may come to the edge of that part (a fir bough's dark edge included).
_DIVIDER_NEED = {"stars": MOTIF_MIN + BALL_RIM, "snowflakes": MOTIF_MIN + BALL_RIM,
                 "beads": MOTIF_MIN + BALL_RIM, "candy": 1.9 + BALL_RIM, "fir": 2.6 + EDGE_MM}
_RING_NEED = {"stars": MOTIF_MIN + BALL_RIM, "snowflakes": MOTIF_MIN + BALL_RIM,
              "beads": MOTIF_MIN + BALL_RIM, "zigzag": 1.65 + BALL_RIM, "fir": 2.4 + EDGE_MM}


def divider_width_needed(kind: str) -> float:
    """The narrowest a divider can be, in mm at the middle of the ball, and
    still hold this decoration."""
    return 2 * (_DIVIDER_NEED.get(kind, 0.0) + BALL_EDGE)


def ring_height_needed(kind: str) -> float:
    """The least height, in mm, of plain ball above or below the pictures
    that holds this decoration."""
    return 2 * (_RING_NEED.get(kind, 0.0) + BALL_EDGE)


def divider_half(lay: Layout) -> float:
    """Half the width of a divider between two pictures, radians of longitude."""
    return max(0.0, 0.5 * lay.sector - lay.panel_half)


def divider_centres(lay: Layout, back: bool = False) -> list[float]:
    """Longitudes of the middles of the dividers - without the one at the
    back, which carries the seam and stays plain, unless back is set."""
    if divider_half(lay) <= 1e-9:
        return []
    centres = [(c + 0.5 * lay.sector) % (2 * math.pi) for c in lay.centers]
    return [c for c in centres if back or abs(wrap_angle(c - BACK_LON)) > 1e-6]


def _motif(sheet: Sheet, kind: str, at: tuple[float, float], size: float, k: int) -> None:
    """One star, snowflake or bead, `size` mm from its middle to its tips."""
    if kind == "stars":
        _star(sheet, at, size, 0.0, edge=BALL_RIM)
    elif kind == "snowflakes":
        _snowflake(sheet, at, size, (k % 2) * math.pi / 6, edge=BALL_RIM)
    else:
        sheet.disc(at, 0.6 * size, "light", edge=BALL_RIM)


def _motif_pitch(kind: str, size: float) -> float:
    """Distance between the middles of two neighbouring motifs."""
    return (1.2 * size + 2.0) if kind == "beads" else (2.0 * size + 2.0)


def _candy_strip(sheet: Sheet, lon, lat, along, width) -> None:
    """A candy-cane stick along an open path: `width` mm across (one value
    per point of the path)."""
    width = np.broadcast_to(np.asarray(width, dtype=np.float64), np.shape(along))
    if BALL_RIM > 0.0:
        sheet.ribbon(lon, lat, width + 2 * BALL_RIM, "dark")
    sheet.ribbon(lon, lat, width, "light")
    total = float(along[-1])
    count = max(2, int(round(total / 7.6)))
    pitch = total / count
    share = np.linspace(-1.0, 1.0, 9)                       # from one edge of the band to the other
    for shift, wide in ((0.0, 2.5), (0.5 * pitch + 0.1, 1.15)):
        for k in range(count):
            centre = (k + 0.25) * pitch + shift
            slant = 0.5 * float(np.interp(min(centre, total), along, width))    # 45 degrees here
            edge_lon, edge_lat = [], []
            for where, part in ((centre + share * slant - wide / 2, share),
                                (centre + share[::-1] * slant + wide / 2, share[::-1])):
                inside = np.clip(where, 0.0, total)
                p_lon, p_lat, heading = _at(lon, lat, along, inside)
                # every point as far out as the band is wide THERE (the band narrows towards
                # its ends); past either end the stripe is cut off square
                b = part * (0.5 * np.interp(inside, along, width) + 0.5 * BALL_RIM)
                b = np.where((where < 0.0) | (where > total), 0.0, b)
                e_lon, e_lat = sheet.offset(p_lon, p_lat, -np.sin(heading) * b, np.cos(heading) * b)
                edge_lon.append(e_lon)
                edge_lat.append(e_lat)
            sheet.fill(np.concatenate(edge_lon), np.concatenate(edge_lat), "dark")


def _draw_divider(sheet: Sheet, kind: str) -> bool:
    """The decoration of one divider, drawn up its middle.  False if the
    divider is too narrow to hold it."""
    panel, radius = sheet.panel, sheet.panel.radius
    tall = panel.band_half * radius - 1.6                   # half the height there is to fill

    def room(phi):                                          # mm from the middle line to the limit
        return panel.panel_half * radius * np.cos(phi) - BALL_EDGE

    if kind not in _DIVIDER_NEED or float(room(0.0)) < _DIVIDER_NEED[kind] - 1e-9:
        return False
    if kind in ("stars", "snowflakes", "beads"):
        size = min(MOTIF_MAX, float(room(0.0)) - BALL_RIM)
        pitch = _motif_pitch(kind, size)
        count = max(1, int((2 * tall - 2 * size) // pitch) + 1)
        for k in range(count):
            phi = (k - 0.5 * (count - 1)) * pitch / radius
            here = min(size, float(room(phi)) - BALL_RIM)
            if here >= 0.85 * MOTIF_MIN:
                _motif(sheet, kind, (0.0, phi), here, k)
        return True
    # A band up the middle, as wide as the divider leaves room for at each height.  It ends
    # where the divider gets too narrow for it, and in any case short of the pictures' top
    # and bottom by as much as its own rounded end sticks out.
    rim = BALL_RIM if kind == "candy" else EDGE_MM
    least = 1.2                                             # half the narrowest band worth drawing
    cos_end = (least + rim + BALL_EDGE) / (panel.panel_half * radius)
    tall = min(tall, radius * math.acos(min(1.0, cos_end)))
    end = max(1.0, tall - (float(room(tall / radius)) + (0.0 if kind == "candy" else 0.4)))
    phi = np.linspace(-end / radius, end / radius, max(8, int(2 * end / 0.35)))
    lon = np.zeros_like(phi)
    along = (phi - phi[0]) * radius
    # (no wider than a stick of candy or a bough is: on a very wide divider it runs up the
    # middle with plain ball either side)
    half = np.minimum(np.maximum(room(phi) - rim, least), 3.4 if kind == "candy" else FIR_HALF)
    if kind == "candy":
        _candy_strip(sheet, lon, phi, along, 2 * half)
        return True
    if kind == "fir":
        _fir(sheet, lon, phi, along, 0.95, half=half, seed=17)
        if float(half.max()) >= 3.0:
            spots = max(2, int(round(2 * tall / 15.0)))
            for k in range(spots):
                at = (0.0, float(phi[0] + (k + 0.5) / spots * (phi[-1] - phi[0])))
                sheet.disc(at, min(1.7, 0.5 * float(room(at[1]))), "light", edge=0.8)
        return True
    return False


def _draw_ring(sheet: Sheet, kind: str, lo: float, hi: float, n_images: int) -> bool:
    """A decoration right round the ball between the latitudes lo and hi
    (radians), with a gap at the back.  The sheet is laid out in longitude
    (measured from the front) and latitude.  False if the strip is too low
    to hold it."""
    radius = sheet.panel.radius
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * radius - BALL_EDGE             # mm either side of the middle parallel
    per_rad = radius * math.cos(mid)                        # mm of that parallel per radian
    around = 2 * math.pi * per_rad
    if kind not in _RING_NEED or half < _RING_NEED[kind] - 1e-9:
        return False
    # The gap at the back is measured where the strip is narrowest round the ball - its edge
    # nearer the pole, where the meridians have closed in - so that it is kept everywhere.
    per_back = radius * math.cos(max(abs(lo), abs(hi)))
    if kind in ("stars", "snowflakes", "beads"):
        size = min(MOTIF_MAX, half - BALL_RIM)
        own = (0.6 * size if kind == "beads" else size) + BALL_RIM      # from its middle to its edge
        first = math.pi - (RING_BACK_GAP + own) / per_back  # the nearest a middle may be to the back
        fit = 2 * first * per_rad / _motif_pitch(kind, size) + 1
        # as many as fit - a whole number for each picture, where there is room for that
        count = n_images * int(fit // n_images) if fit >= n_images else max(1, int(fit))
        for k in range(count):                              # evenly, from one side of the back to the other
            dlam = -first + 2 * first * k / max(1, count - 1) if count > 1 else 0.0
            _motif(sheet, kind, (dlam, mid), size, k)
        return True
    edge = math.pi - RING_BACK_GAP / per_back               # where the line stops, either side of the back
    if kind == "zigzag":
        stroke = 1.3
        swing = half - BALL_RIM - stroke / 2                # how far the line swings either side
        periods = max(n_images, n_images * int(round(around / (3.2 * swing + 3.0) / n_images)))
        edge -= (stroke / 2 + BALL_RIM) / per_back          # (its rounded ends stick out that far)
        corners = np.linspace(-edge, edge, 2 * periods + 1)
        heights = np.where(np.arange(2 * periods + 1) % 2 == 0, -swing, swing) / radius + mid
        lon = np.linspace(-edge, edge, max(16, int(around / 0.3)))
        lat = np.interp(lon, corners, heights)
        if BALL_RIM > 0.0:
            sheet.ribbon(lon, lat, stroke + 2 * BALL_RIM, "dark")
        sheet.ribbon(lon, lat, stroke, "light")
        return True
    if kind == "fir":
        reach = min(FIR_HALF, half - EDGE_MM)
        edge -= (reach + EDGE_MM + 0.4) / per_back          # (needles and the bough's end reach that far)
        lon = np.linspace(-edge, edge, max(16, int(around / 0.35)))
        lat = np.full_like(lon, mid)
        along = (lon - lon[0]) * per_rad
        _fir(sheet, lon, lat, along, 0.95, half=reach, seed=19)
        if reach >= 2.8:
            spots = max(n_images, n_images * int(round(around / 13.0 / n_images)))
            for k in range(spots):
                at = (float(-edge + (k + 0.5) / spots * 2 * edge), mid)
                sheet.disc(at, min(BAUBLE, 0.6 * reach), "light", edge=0.8)
        return True
    return False


def _draw_pole(sheet: Sheet, kind: str, reach: float) -> bool:
    """A big snowflake or star on the closed end, drawn about the south pole
    and reaching `reach` mm from it.  No arm of it points at the back, where
    the seam runs up to the pole.  False if there is no room."""
    if reach < 6.0:
        return False
    at = (0.0, -0.5 * math.pi)          # (the sheet's longitude is measured from the back)

    def put(distance, sideways, angle):
        """A polygon given along and across a line that leaves the pole at
        `angle` radians from the back."""
        distance, sideways = np.asarray(distance, dtype=np.float64), np.asarray(sideways, dtype=np.float64)
        east = distance * math.sin(angle) + sideways * math.cos(angle)
        north = distance * math.cos(angle) - sideways * math.sin(angle)
        sheet.shape(at, east, north, "dark")

    with sheet.group("dark"):
        if kind == "snowflake":
            wide = min(2.6, max(1.7, 0.12 * reach))
            for k in range(6):
                angle = math.radians(30.0 + 60.0 * k)
                put([0.0, reach - wide, reach, reach - wide, 0.0],
                    [wide / 2, wide / 2, 0.0, -wide / 2, -wide / 2], angle)
                for share, length in ((0.44, 0.30), (0.70, 0.20)):      # two pairs of branches
                    for side in (-1.0, 1.0):
                        turn = angle + side * math.radians(60.0)
                        root = share * reach
                        # a branch: a bar that leaves the arm at `root`, at 60 degrees to it
                        d = np.array([0.0, length * reach - 0.8 * wide, length * reach,
                                      length * reach - 0.8 * wide, 0.0])
                        w = 0.8 * np.array([wide / 2, wide / 2, 0.0, -wide / 2, -wide / 2])
                        e0, n0 = root * math.sin(angle), root * math.cos(angle)
                        east = e0 + d * math.sin(turn) + w * math.cos(turn)
                        north = n0 + d * math.cos(turn) - w * math.sin(turn)
                        sheet.shape(at, east, north, "dark")
            sheet.disc(at, 1.45 * wide, "dark")
        else:                           # a five-pointed star
            k = np.arange(10)
            r = np.where(k % 2 == 0, reach, 0.42 * reach)
            angle = math.radians(36.0) + k * math.pi / 5
            sheet.shape(at, r * np.sin(angle), r * np.cos(angle), "dark")
    return True


@dataclass
class Ball:
    """What is drawn on the plain parts of an ornament (see ball_for)."""

    dividers: list[tuple[float, float, Layer]] = field(default_factory=list)
    #           (longitude of the middle of a divider, half its width, what is drawn on it)
    top: tuple[float, float, Layer] | None = None       # (from latitude, to latitude, the ring)
    bottom: tuple[float, float, Layer] | None = None
    pole: tuple[float, Layer] | None = None             # (up to latitude, the shape on the closed end)
    no_room: list[str] = field(default_factory=list)    # settings asked for that found no room

    @property
    def empty(self) -> bool:
        return not (self.dividers or self.top or self.bottom or self.pole)


_balls: dict[tuple, Ball] = {}


def ball_strips(lay: Layout) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Where the plain parts above and below the pictures are: (latitudes of
    the strip above, of the strip below, the latitude the closed end reaches
    up to).  The wall is a plain sphere, inside and out, over both strips; on
    the closed end only the outside is."""
    return (lay.band_half, lay.phi_t), (-lay.phi_c, -lay.band_half), -lay.phi_c


def ball_for(p: Params, lay: Layout, pitch_mm: float) -> Ball | None:
    """The decorations on the plain parts of the ball, drawn finely enough for
    a model sampled every pitch_mm (cached), or None if none are asked for."""
    if not p.ball_decorated:
        return None
    key = (p.ball_dividers, p.ball_top, p.ball_bottom, p.ball_pole, lay.n_images,
           round(lay.R, 4), round(lay.sector, 6), round(lay.panel_half, 6), round(lay.band_half, 6),
           round(lay.phi_t, 6), round(lay.phi_c, 6), round(pitch_mm, 4), ring_below(p))
    with _layers_lock:
        hit = _balls.get(key)
    if hit is None:
        hit = _render_ball(p, lay, pitch_mm)
        with _layers_lock:
            while len(_balls) >= 6:
                _balls.pop(next(iter(_balls)))
            _balls[key] = hit
    return hit


def _render_ball(p: Params, lay: Layout, pitch_mm: float) -> Ball:
    ball = Ball()
    scale = SUPERSAMPLE / max(pitch_mm, 0.05)
    above, below, closed = ball_strips(lay)
    if p.ball_dividers != "none":
        centres = divider_centres(lay)
        sheet = Sheet(Panel(lay.R, divider_half(lay), lay.band_half), scale) if centres else None
        if sheet is not None and _draw_divider(sheet, p.ball_dividers):
            layer = Layer.of(sheet)                         # every divider carries the same
            ball.dividers = [(c, divider_half(lay), layer) for c in centres]
        else:
            ball.no_room.append("ball_dividers")
    for name, (lo, hi) in (("ball_top", above), ("ball_bottom", below)):
        kind = getattr(p, name)
        if kind == "none":
            continue
        if name == "ball_bottom" and not ring_below(p):
            ball.no_room.append(name)
            continue
        sheet = Sheet(Panel(lay.R, math.pi, 0.5 * (hi - lo)), scale, projection="wrap",
                      window=(math.pi, hi, lo)) if hi - lo > 1e-6 else None
        if sheet is not None and _draw_ring(sheet, kind, lo, hi, lay.n_images):
            setattr(ball, name[5:], (lo, hi, Layer.of(sheet)))
        else:
            ball.no_room.append(name)
    if p.ball_pole != "none":
        cap = closed + 0.5 * math.pi                        # the closed end, radians from the pole
        sheet = Sheet(Panel(lay.R, math.pi, cap), scale, projection="polar", window=(cap, cap, -cap))
        if _draw_pole(sheet, p.ball_pole, cap * lay.R - POLE_EDGE):
            ball.pole = (closed, Layer.of(sheet))
        else:
            ball.no_room.append("ball_pole")
    return ball


def ring_below(p: Params) -> bool:
    """True if a ring of decoration can be made round the bottom of the ball.

    Not with the relief on the inside of the wall.  The strip below the
    pictures is then where the inside runs in to the closed end: it is a
    ceiling there, a pattern on it could only be built with edges eased over
    most of the strip, and those would reach the row where the inside joins
    the closing cone."""
    return p.relief_side != "inside"


def ball_reach(p: Params, lay: Layout) -> tuple[float, float]:
    """The latitudes between which the wall carries relief: the pictures, and
    the strips above and below them if those are decorated."""
    above, below, _closed = ball_strips(lay)
    return (below[0] if p.ball_bottom != "none" and ring_below(p) else -lay.band_half,
            above[1] if p.ball_top != "none" else lay.band_half)


def apply_ball(T: np.ndarray, phis: np.ndarray, lam: np.ndarray, p: Params, lay: Layout,
               pitch_mm: float, parts: tuple[str, ...] = ("dividers", "top", "bottom")) -> None:
    """Lay the decorations of the plain parts over the wall thickness T (rows
    = latitudes phis, columns = longitudes lam), in place.  Pictures are not
    touched.  parts: which of them - the dividers, the strip above the
    pictures, the strip below."""
    ball = ball_for(p, lay, pitch_mm)
    if ball is None or ball.empty:
        return
    in_band = np.nonzero(np.abs(phis) < lay.band_half)[0]
    for centre, half, layer in (ball.dividers if "dividers" in parts else ()):
        dlam = wrap_angle(lam - centre)
        cols = np.nonzero(np.abs(dlam) < half)[0]
        if cols.size and in_band.size:
            where = np.ix_(in_band, cols)
            dl, ph = np.meshgrid(dlam[cols], phis[in_band])
            T[where] = apply(T[where], layer, dl, ph, p)
    dlam = wrap_angle(lam - FRONT_LON)
    for name, strip in (("top", ball.top), ("bottom", ball.bottom)):
        if strip is None or name not in parts:
            continue
        lo, hi, layer = strip
        rows = np.nonzero((phis > lo) & (phis < hi))[0]
        if rows.size:
            dl, ph = np.meshgrid(dlam, phis[rows])
            T[rows] = apply(T[rows], layer, dl, ph, p)


def pole_relief(p: Params, lay: Layout, phis: np.ndarray, lam: np.ndarray,
                pitch_mm: float) -> np.ndarray | None:
    """How far the outside of the closed end stands proud of the ball, in mm,
    at latitudes phis (all below the closed latitude) x longitudes lam - or
    None if nothing is drawn there."""
    ball = ball_for(p, lay, pitch_mm)
    if ball is None or ball.pole is None:
        return None
    _closed, layer = ball.pole
    dl, ph = np.meshgrid(wrap_angle(lam - BACK_LON), phis)
    hidden, light, plain = layer.sample(dl, ph)
    return POLE_RAISE * np.clip(hidden - light - plain, 0.0, 1.0).astype(np.float64)


def backdrop_thickness(slot: ImageSlot, p: Params) -> float:
    """Wall thickness of a picture that has no photo, only lettering or decorations."""
    return {"dark": p.t_max, "light": p.t_min}.get(slot.backdrop, p.t_frame)
