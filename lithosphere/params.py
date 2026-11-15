"""Parameters, limits, validation and derived geometry for the ornament.

Everything is in millimetres and degrees on the user-facing side.

Coordinate convention used throughout the package ("hanging" frame):
    origin  = centre of the sphere
    +Z      = up, towards the neck / hanger
    lat     = latitude, +90 deg at the neck, -90 deg at the bottom of the ornament
    lon     = longitude, counter-clockwise seen from above; an image's left edge
              is at the smaller longitude, so nothing is mirrored from outside.

The printed part is this shape turned upside-down (neck on the bed).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, NamedTuple

MAX_IMAGES = 4

# Printer assumptions (Bambu Lab A1).
PRINTER_NAME = "Bambu Lab A1"
BED_SIZE_MM = 256.0
MAX_PRINT_HEIGHT_MM = 256.0
PLA_DENSITY_G_CM3 = 1.24


@dataclass(frozen=True)
class Nozzle:
    """What the app has to know about one nozzle size of the A1."""

    label: str
    diameter: float                 # mm
    line_width: float               # mm - Bambu Studio's line width for this nozzle
    layers: tuple[float, ...]       # lithophane-grade layer heights, tallest first
    machine: str                    # Bambu Studio system presets ...
    filament: str
    processes: tuple[str, ...]      # ... one process preset per entry of `layers`
    pitches: dict[str, float]       # model detail -> sample spacing on the surface, mm
    hours_per_cm3: float            # print time per cm3 of plastic at the tallest layer


# The 0.2 mm nozzle lays lines about half as wide (0.22 instead of 0.42 mm).  Its
# layers are half as tall as well, so every overhang in the design - shoulder,
# closed end, picture relief - steps sideways by the same share of a line as it
# does with the 0.4 mm nozzle, where the design is proven.  The model is sampled
# finer to match.  A line with a quarter of the cross-section needs four times
# the length for the same plastic, hence the print time.
NOZZLES: dict[str, Nozzle] = {
    "0.4": Nozzle("0.4 mm", 0.4, 0.42, (0.12, 0.08),
                  "Bambu Lab A1 0.4 nozzle", "Bambu PLA Basic @BBL A1",
                  ("0.12mm High Quality @BBL A1", "0.08mm High Quality @BBL A1"),
                  {"draft": 0.40, "standard": 0.25, "fine": 0.18}, 0.128),
    "0.2": Nozzle("0.2 mm", 0.2, 0.22, (0.06,),
                  "Bambu Lab A1 0.2 nozzle", "Bambu PLA Basic @BBL A1 0.2 nozzle",
                  ("0.06mm High Quality @BBL A1 0.2 nozzle",),
                  {"draft": 0.25, "standard": 0.15, "fine": 0.12}, 0.38),
}
DEFAULT_NOZZLE = "0.4"


class LightPreset(NamedTuple):
    """A light the cap can be made for.

    hole: diameter of the opening (None = whatever the user sets).  seat: for a
    light that sits ON the cap and shines down through the opening, the
    diameter of the pocket it drops into; depth: how deep that pocket is."""

    label: str
    hole: float | None
    seat: float | None = None
    depth: float | None = None


LIGHT_PRESETS: dict[str, LightPreset] = {
    "mini": LightPreset("Mini string light (5 mm LED / mini bulb)", 10.0),
    "fairy": LightPreset("Fairy lights (micro LEDs on wire)", 6.0),
    # battery "balloon" / paper-lantern lights: the barrel measures 11.0 - 11.2 mm and the
    # silicone ring behind it stops on the cap; makers of lithophane balls use 11.4 - 11.6 mm
    "balloon": LightPreset("LED balloon light (battery, 11 mm bullet)", 11.6),
    # "submersible" LED tea lights: a puck with two CR2032 cells that is switched by twisting
    # it.  It sits in a pocket in the top of the cap and shines down into the ball.  Sold as
    # 30 mm; the makes that give millimetres say 28.5 - 30.0 across and 19 - 26 tall, and
    # people who measured theirs found 29.0 - 29.6.  The pocket takes the largest, and its
    # conical floor centres the smaller ones over an opening that none of them can fall through.
    "puck": LightPreset("Puck light, 30 mm (sits in the cap)", 24.5, 30.6, 10.0),
    # LED tea lights, flame down: bodies measure 35.7 - 38.1 mm across (one CR2032, or a CR2450
    # in the kind with a remote and a timer), the flame is about a centimetre wide
    "tealight": LightPreset("LED tea light, 38 mm (sits in the cap)", 32.0, 39.0, 12.0),
    "c6": LightPreset("C6 LED bulb (19 mm)", 20.5),
    "c7": LightPreset("C7 LED bulb (22 mm)", 24.0),
    "c9": LightPreset("C9 LED bulb (29 mm)", 30.5),
    "custom": LightPreset("Custom size", None),
}

# name -> label.  How finely the model is sampled depends on the nozzle: see mesh_pitch().
QUALITY_PRESETS: dict[str, str] = {
    "draft": "Draft  - small file, quick test",
    "standard": "Standard  - recommended",
    "fine": "Fine  - maximum detail, big file",
}

PROJECTIONS = {
    "natural": "Natural - photo keeps its proportions (corners trimmed)",
    "wrap": "Wrap - whole photo kept (narrows towards top and bottom)",
}
RELIEF_SIDES = {
    # (the side panel has room for about 60 characters)
    "outside": "Outside - sharpest picture when lit (recommended)",
    "split": "Both - smooth upper half, but its picture looks soft",
    "inside": "Inside - smooth all over, but the picture looks soft",
}
HANGERS = {
    "two": "Two loops (ribbon or hook, hangs straight)",
    "one": "One loop",
    "none": "None (sits on the light / branch)",
}

# Lettering and decorations that can be laid over a picture (see overlay.py).
TEXT_TONES = {
    "light": "Glowing letters",
    "dark": "Dark letters",
}
BACKDROPS = {               # what a picture without a photo is made of
    "plain": "Plain",
    "dark": "Dark",
    "light": "Bright",
}
DECO_TOP = {
    "none": "Nothing",
    "garland": "Fir garland with baubles and bows",
    "lights": "String of fairy lights",
    "icicles": "Snow and icicles",
    "holly": "Holly in the corners",
}
DECO_BOTTOM = {
    "none": "Nothing",
    "garland": "Fir garland with baubles and bows",
    "snow": "Bank of snow",
    "trees": "Snow with fir trees",
    "holly": "Holly in the corners",
}
DECO_FRAME = {
    "none": "Nothing",
    "line": "Line frame",
    "candy": "Candy-cane border",
    "wreath": "Round wreath",
    "wreath_square": "Wreath along the edges",
}
DECO_SCATTER = {
    "none": "Nothing",
    "snowflakes": "Snowflakes",
    "snow": "Falling snow",
    "stars": "Stars",
    "hearts": "Hearts",
}
MAX_TEXT_CHARS = 240

# Decorations on the plain parts of the ball itself (see overlay.ball_for).
BALL_DIVIDERS = {           # on the dividers between the pictures
    "none": "Nothing",
    "stars": "Stars",
    "snowflakes": "Snowflakes",
    "beads": "String of beads",
    "candy": "Candy-cane stripes",
    "fir": "Fir branch",
}
BALL_RING = {               # round the ball above the pictures, or below them
    "none": "Nothing",
    "stars": "Row of stars",
    "snowflakes": "Row of snowflakes",
    "beads": "String of beads",
    "zigzag": "Zigzag",
    "fir": "Fir garland",
}
BALL_POLE = {               # on the closed end, underneath
    "none": "Nothing",
    "snowflake": "Big snowflake",
    "star": "Big star",
}
BALL_SETTINGS = ("ball_dividers", "ball_top", "ball_bottom", "ball_pole")

# (min, max) limits used by both the UI and validation.
LIMITS: dict[str, tuple[float, float]] = {
    "diameter": (50.0, 150.0),
    "band_height_deg": (40.0, 120.0),
    "gap_deg": (0.0, 40.0),
    "t_min": (0.4, 1.6),
    "t_max": (1.4, 4.0),
    "t_frame": (0.8, 4.0),
    "gamma": (0.5, 2.0),
    "smoothing": (0.0, 3.0),
    "hole_diameter": (4.0, 45.0),
    "seat_diameter": (12.0, 56.0),
    "seat_depth": (3.0, 30.0),
    "cap_diameter": (8.0, 60.0),
    "cap_height": (4.0, 40.0),
    "cap_wall": (1.2, 4.0),
    "flare_angle": (25.0, 50.0),
    "hanger_hole": (2.0, 6.0),
    "zoom": (0.5, 4.0),
    "shift": (-1.0, 1.0),
    "brightness": (-0.5, 0.5),
    "contrast": (0.5, 2.0),
    "text_size": (0.15, 1.0),
    "text_outline": (0.0, 2.5),
    "text_bold": (0.0, 1.0),
    "deco_amount": (0.0, 1.0),
    "deco_size": (0.6, 1.6),
}


@dataclass
class ImageSlot:
    """One photo and how it is placed in its panel."""

    path: str = ""
    zoom: float = 1.0          # 1.0 = photo just covers the panel
    shift_x: float = 0.0       # -1..1, slides the photo within the spare margin
    shift_y: float = 0.0
    rotation: int = 0          # 0 / 90 / 180 / 270, clockwise
    flip: bool = False         # mirror left-right
    brightness: float = 0.0    # -0.5..0.5
    contrast: float = 1.0      # 0.5..2
    auto_levels: bool = True   # stretch the photo's tones to the full range
    # --- lettering over the photo - or instead of one (see overlay.py) ---------
    text: str = ""             # lines separated by "\n"
    text_font: str = ""        # name of an installed font; "" = this computer's default
    text_tone: str = "light"   # "light": letters glow (thin wall); "dark": letters stay dark
    text_size: float = 0.85    # share of the largest size that still fits the picture
    text_x: float = 0.0        # -1..1, slides the lettering left / right in the room it has
    text_y: float = -0.6       # -1..1, down / up (lettering is sharpest below the middle)
    text_outline: float = 1.0  # mm of contrasting edge round every letter
    text_bold: float = 0.0     # mm added to every stroke
    # --- decorations -----------------------------------------------------------
    deco_top: str = "none"     # along the top edge (a key of DECO_TOP)
    deco_bottom: str = "none"
    deco_frame: str = "none"   # all the way round
    deco_scatter: str = "none"  # sprinkled over what is left free
    deco_amount: float = 0.5   # how many scattered shapes, 0..1
    deco_size: float = 1.0     # scale of every decoration
    deco_seed: int = 0         # which arrangement of the scattered shapes
    backdrop: str = "plain"    # what the picture is made of when it has no photo

    def reset_adjustments(self) -> None:
        """Undo the cropping and tone adjustments of the photo (the photo
        itself, the lettering and the decorations stay)."""
        fresh = ImageSlot()
        for name in PHOTO_SETTINGS:
            setattr(self, name, getattr(fresh, name))

    def clear_overlay(self) -> None:
        """Take the lettering and every decoration off this picture."""
        fresh = ImageSlot()
        for name in OVERLAY_SETTINGS:
            setattr(self, name, getattr(fresh, name))

    def keep_overlay_of(self, other: "ImageSlot") -> "ImageSlot":
        """This slot with the lettering and decorations of another one."""
        for name in OVERLAY_SETTINGS:
            setattr(self, name, getattr(other, name))
        return self

    @property
    def decorated(self) -> bool:
        """True if anything is drawn over this picture: lettering or a decoration."""
        return bool(self.text.strip() or any(getattr(self, name) != "none" for name in
                                             ("deco_top", "deco_bottom", "deco_frame", "deco_scatter")))


PHOTO_SETTINGS = ("zoom", "shift_x", "shift_y", "rotation", "flip", "brightness", "contrast",
                  "auto_levels")
DECO_SETTINGS = ("deco_top", "deco_bottom", "deco_frame", "deco_scatter", "deco_amount", "deco_size",
                 "deco_seed")
OVERLAY_SETTINGS = ("text", "text_font", "text_tone", "text_size", "text_x", "text_y",
                    "text_outline", "text_bold") + DECO_SETTINGS + ("backdrop",)


@dataclass
class Params:
    # --- sphere / layout -------------------------------------------------
    diameter: float = 80.0
    n_images: int = 3
    band_height_deg: float = 100.0   # latitude span covered by the pictures
    gap_deg: float = 12.0            # divider between neighbouring pictures (hides the seam)
    projection: str = "natural"
    relief_side: str = "outside"
    # --- lithophane ------------------------------------------------------
    t_min: float = 0.8               # thinnest wall = brightest
    t_max: float = 2.8               # thickest wall = darkest
    t_frame: float = 2.2             # dividers, caps and neck
    gamma: float = 1.0               # >1 lifts mid-tones
    smoothing: float = 0.5           # blur, in mesh samples
    invert: bool = False
    safe_edges: bool = True          # ramp edges the printer could not build over thin air
    # --- neck, light and hanger -----------------------------------------
    light_preset: str = "mini"
    hole_diameter: float = 10.0      # opening for the light
    seat: bool = False               # a pocket in the top of the cap for a light that sits on it
    seat_diameter: float = 30.6      # ... its diameter (the light's own, plus clearance)
    seat_depth: float = 10.0         # ... and how deep it is
    cap_diameter: float = 26.0       # the cylindrical cap on top of the sphere
    cap_height: float = 9.0
    cap_wall: float = 2.0            # cap wall and top-plate thickness
    flare_angle: float = 30.0        # shoulder cone angle, from horizontal
    hanger: str = "two"
    hanger_hole: float = 3.5
    # --- decorations on the plain parts of the ball (see overlay.ball_for) --
    ball_dividers: str = "none"      # between the pictures (a key of BALL_DIVIDERS)
    ball_top: str = "none"           # round the ball above the pictures (BALL_RING)
    ball_bottom: str = "none"        # ... and below them (BALL_RING)
    ball_pole: str = "none"          # underneath, on the closed end (BALL_POLE)
    # --- output ----------------------------------------------------------
    quality: str = "standard"
    nozzle: str = DEFAULT_NOZZLE     # nozzle the ornament is printed with (a key of NOZZLES)
    images: list[ImageSlot] = field(default_factory=lambda: [ImageSlot() for _ in range(MAX_IMAGES)])

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Params":
        p = cls()
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key == "images" or key not in known:
                continue
            current = getattr(p, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, str):
                    value = str(value)
            except (TypeError, ValueError, OverflowError):     # (int(inf), float of a huge int)
                continue
            setattr(p, key, value)
        slot_fields = {f.name for f in fields(ImageSlot)}
        slots = []
        listed = data.get("images")
        for raw in (listed if isinstance(listed, list) else [])[:MAX_IMAGES]:
            if isinstance(raw, dict):
                slots.append(ImageSlot(**{k: v for k, v in raw.items() if k in slot_fields}))
        while len(slots) < MAX_IMAGES:
            slots.append(ImageSlot())
        p.images = slots
        p.sanitize()
        return p

    @classmethod
    def from_json(cls, text: str) -> "Params":
        return cls.from_dict(json.loads(text))

    def copy(self) -> "Params":
        return Params.from_dict(self.to_dict())

    @property
    def ball_decorated(self) -> bool:
        """True if anything is asked for on the plain parts of the ball."""
        return any(getattr(self, name) != "none" for name in BALL_SETTINGS)

    # -- cleaning -------------------------------------------------------------
    def sanitize(self) -> None:
        """Clamp every value into its legal range (never raises)."""

        def clamp(name: str, value: float) -> float:
            lo, hi = LIMITS[name]
            if not math.isfinite(value):
                return lo
            return min(hi, max(lo, value))

        def number(value: Any, fallback: float) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                return fallback
            return value if math.isfinite(value) else fallback

        for name in ("diameter", "band_height_deg", "gap_deg", "t_min", "t_max", "t_frame",
                     "gamma", "smoothing", "hole_diameter", "seat_diameter", "seat_depth",
                     "cap_diameter", "cap_height", "cap_wall", "flare_angle", "hanger_hole"):
            setattr(self, name, clamp(name, float(getattr(self, name))))
        self.n_images = int(min(MAX_IMAGES, max(1, self.n_images)))
        if self.nozzle not in NOZZLES:
            self.nozzle = DEFAULT_NOZZLE
        if self.projection not in PROJECTIONS:
            self.projection = "natural"
        if self.relief_side not in RELIEF_SIDES:
            self.relief_side = "outside"
        if self.hanger not in HANGERS:
            self.hanger = "two"
        if self.quality not in QUALITY_PRESETS:
            self.quality = "standard"
        if self.light_preset not in LIGHT_PRESETS:
            self.light_preset = "custom"
        for name, choices in (("ball_dividers", BALL_DIVIDERS), ("ball_top", BALL_RING),
                              ("ball_bottom", BALL_RING), ("ball_pole", BALL_POLE)):
            value = getattr(self, name)
            if not isinstance(value, str) or value not in choices:
                setattr(self, name, "none")
        for slot in self.images:
            slot.zoom = clamp("zoom", float(slot.zoom))
            slot.shift_x = clamp("shift", float(slot.shift_x))
            slot.shift_y = clamp("shift", float(slot.shift_y))
            slot.brightness = clamp("brightness", float(slot.brightness))
            slot.contrast = clamp("contrast", float(slot.contrast))
            slot.rotation = int(slot.rotation) % 360 // 90 * 90
            slot.flip = bool(slot.flip)
            slot.auto_levels = bool(slot.auto_levels)
            slot.path = str(slot.path or "")
            slot.text = str(slot.text or "").replace("\r", "")[:MAX_TEXT_CHARS]
            slot.text_font = str(slot.text_font or "")
            slot.text_size = clamp("text_size", number(slot.text_size, 0.85))
            slot.text_x = clamp("shift", number(slot.text_x, 0.0))
            slot.text_y = clamp("shift", number(slot.text_y, -0.6))
            slot.text_outline = clamp("text_outline", number(slot.text_outline, 1.0))
            slot.text_bold = clamp("text_bold", number(slot.text_bold, 0.0))
            slot.deco_amount = clamp("deco_amount", number(slot.deco_amount, 0.5))
            slot.deco_size = clamp("deco_size", number(slot.deco_size, 1.0))
            slot.deco_seed = int(number(slot.deco_seed, 0.0)) % 10000
            for name, choices in (("text_tone", TEXT_TONES), ("backdrop", BACKDROPS),
                                  ("deco_top", DECO_TOP), ("deco_bottom", DECO_BOTTOM),
                                  ("deco_frame", DECO_FRAME), ("deco_scatter", DECO_SCATTER)):
                value = getattr(slot, name)
                if not isinstance(value, str) or value not in choices:
                    setattr(slot, name, next(iter(choices)))


OLD_DEFAULT_GAP_DEG = 6.0       # the default gap between pictures before version 1.2


def params_from_file(data: dict[str, Any]) -> Params:
    """Params from the contents of a project or session file.

    Files written before version 1.1 could only put the relief on one side,
    and "inside" was simply the default.  Version 1.1 and 1.2 moved those to
    "split" (inside above the equator, outside below it) and made that the
    default.  The first ornament printed that way showed what it costs: lit,
    the half with the relief on the inside looks blurred, because the light
    spreads sideways in the plastic after it has passed the relief.  So files
    written before version 1.3 that carry either old default move to
    "outside", which keeps the whole picture sharp.  A file that says
    "outside", or "inside" from version 1.1 on, was a deliberate choice and is
    left alone.

    Files written before version 1.2 that still carry the old default gap of
    6 degrees move to the present default.  Any other gap was chosen on
    purpose and is kept."""
    params = Params.from_dict(data.get("params", data))
    numbers = [int(x) for x in str(data.get("version") or "0").split(".")[:2] if x.isdigit()]
    written_by = tuple(numbers + [0, 0])[:2]
    if written_by < (1, 1) and params.relief_side == "inside":
        params.relief_side = "outside"
    if written_by < (1, 3) and params.relief_side == "split":
        params.relief_side = "outside"
    if written_by < (1, 2) and abs(params.gap_deg - OLD_DEFAULT_GAP_DEG) < 1e-9:
        params.gap_deg = Params().gap_deg
    return params


class GeometryError(ValueError):
    """The parameters describe a shape that cannot be built.

    kind names the rule that was broken, so make_buildable() can repair it."""

    def __init__(self, message: str, kind: str = ""):
        super().__init__(message)
        self.kind = kind


@dataclass
class Layout:
    """Dimensions derived from Params (hanging frame, mm / radians)."""

    R: float                 # nominal outer radius
    ro_frame: float          # outer radius where there is no picture
    ri_frame: float          # inner radius where there is no picture
    r_smooth: float          # radius of the smooth (non-relief) sphere surface
    theta_s: float           # co-latitude from the top where the shoulder cone starts
    phi_t: float             # same thing as a latitude
    rh: float                # light-opening radius
    rc: float                # cap outer radius
    rci: float               # cap inner radius (= rh when the cap is a plain tube)
    cup: bool                # True: hollow cap with a top plate around the light opening
    z_plate: float           # height of the underside of that plate (see derive)
    seat: bool               # True: the top of the cap has a pocket for a light to sit in
    r_seat: float            # radius of the pocket (0 without one)
    z_seat: float            # height of the pocket's floor at its wall ...
    z_bore: float            # ... which runs down as a cone to the opening at this height
    r_to: float              # outer shoulder / sphere tangent circle radius
    z_to: float
    z_oc: float              # height where the outer shoulder meets the cap wall
    z_top: float             # top face of the cap (rests on the bed when printing)
    r_ti: float              # inner shoulder / sphere tangent circle radius
    z_ti: float
    z_ic: float              # height where the inner shoulder meets the cap's inner wall
    phi_c: float             # the inside follows the sphere down to latitude -phi_c ...
    r_f: float               # ... then a cone, then a flat bridged ceiling of this radius
    z_f: float               # depth of that ceiling below the centre
    roof: float              # thinnest plastic over the closed end
    band_half: float         # half latitude span of the pictures
    panel_half: float        # half longitude span of one picture
    sector: float            # longitude span of one panel incl. divider
    centers: list[float]     # panel centre longitudes
    projection: str          # projection actually used
    height: float            # printed height
    width: float             # printed width incl. hanger loops
    notes: list[str] = field(default_factory=list)   # non-fatal adjustments

    @property
    def n_images(self) -> int:
        return len(self.centers)


# Longitude that faces the back of the printer (+Y). A divider is placed here so
# the slicer's "Back" seam lands between two pictures.
BACK_LON = math.pi / 2


def hanger_dims(p: Params, rc: float) -> dict[str, float]:
    """Dimensions of one hanger loop (a flat tab at the top edge of the cap)."""
    hole_r = p.hanger_hole / 2
    rim = 2.2                               # plastic around the hole
    thickness = min(2.4, p.cap_height)      # tab thickness
    cx = rc + hole_r + 0.8                  # hole centre, just outside the cap
    return {"hole_r": hole_r, "half_w": hole_r + rim, "thickness": thickness, "cx": cx,
            "reach": cx + hole_r + rim}


def _frame_radii(p: Params) -> tuple[float, float, float]:
    """(outer, inner, smooth-surface) sphere radii where there is no picture.

    "inside" and "split" keep the frame on the nominal sphere ("split" then
    lets the darkest parts of the lower pictures stand proud of it);
    "outside" keeps the darkest parts on the nominal sphere instead."""
    R = p.diameter / 2
    if p.relief_side != "outside":
        return R, R - p.t_frame, R
    ri = R - p.t_max
    return ri + p.t_frame, ri, ri


MIN_CAP_WALL = 1.2      # least plastic between the light opening and the cap's outside

# A light that sits ON the cap (a battery puck) drops into a pocket in the top
# of the cap and rests on a ledge round the opening it shines through.  Printed
# cap-down that ledge is a ceiling over the pocket, so it is a cone instead of a
# flat step: it prints without support and centres the light.
SEAT_LEDGE_MIN = 1.5    # least ledge (mm) all round the opening
SEAT_CONE_DEG = 45.0    # slope of the ledge, from flat
SEAT_NECK_MM = 0.8      # straight part of the opening below the cone
SEAT_CLEARANCE_MM = 0.6     # a pocket this much wider than the light lets it drop in freely


def seat_opening_max(p: Params) -> float:
    """Widest light opening that still leaves the ledge under a seated light."""
    return p.seat_diameter - 2 * SEAT_LEDGE_MIN


def seat_stack_mm(p: Params) -> float:
    """Height the seat takes up in the cap: pocket, ledge cone and neck."""
    ledge = max(0.0, (p.seat_diameter - p.hole_diameter) / 2)
    return p.seat_depth + ledge * math.tan(math.radians(SEAT_CONE_DEG)) + SEAT_NECK_MM

# The closed end of the ornament is printed last, so its inside is a ceiling over
# empty air.  A plain dome gets flatter and flatter there and prints badly, so
# instead the inside leaves the sphere at this angle (from horizontal), carries
# on as a straight cone and finishes with a small flat ceiling that the printer
# bridges.  The outside stays a perfect sphere.
CLOSURE_ANGLE_DEG = 30.0
CLOSURE_ROOF_MM = 1.2   # least roof thickness at the rim of the flat ceiling


def closure_profile(ro: float, ri: float) -> tuple[float, float, float, float]:
    """(theta_c, r_c, r_f, z_f) of the closed end for outer / inner radii ro, ri.

    theta_c: co-latitude from the closed pole where the cone starts
    r_c:     radius of that tangent circle
    r_f:     radius of the flat ceiling (0 if the cone closes to a point)
    z_f:     distance of the ceiling (or cone tip) from the sphere centre
    """
    theta_c = math.radians(CLOSURE_ANGLE_DEG)
    slope = math.tan(theta_c)
    r_c, z_c = ri * math.sin(theta_c), ri * math.cos(theta_c)

    def roof(r: float) -> float:            # vertical plastic above the cone at radius r
        return math.sqrt(ro * ro - r * r) - (z_c + (r_c - r) * slope)

    target = min(CLOSURE_ROOF_MM, max(0.6, 0.62 * roof(r_c)))
    if roof(0.0) >= target:                 # thick shell: the cone closes inside the plastic
        return theta_c, r_c, 0.0, z_c + r_c * slope
    lo, hi = 0.0, r_c                       # roof() grows with r: bisect for roof(r) = target
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if roof(mid) < target:
            lo = mid
        else:
            hi = mid
    return theta_c, r_c, hi, z_c + (r_c - hi) * slope


def cap_diameter_choices(p: Params) -> tuple[float, float]:
    """cap_diameter_range() rounded inwards to the 0.1 mm the controls show, so
    a value clamped to either end is still a tidy number.  (lo > hi means no
    cap fits at all.)"""
    lo, hi = cap_diameter_range(p)
    if hi < lo:
        return lo, hi
    tidy_lo, tidy_hi = math.ceil(lo * 10 - 1e-9) / 10, math.floor(hi * 10 + 1e-9) / 10
    if tidy_lo > tidy_hi:                   # the whole range is narrower than 0.1 mm
        return lo, hi
    return tidy_lo, tidy_hi


def _closure(ro: float, ri: float) -> tuple[float, float, float]:
    theta_c, _r_c, r_f, z_f = closure_profile(ro, ri)
    return math.pi / 2 - theta_c, r_f, z_f


def closure_roof_mm(ro: float, r_f: float, z_f: float) -> float:
    """Thinnest plastic over the closed end, measured square to the surface
    (it is at the rim of the flat ceiling, or at the cone tip)."""
    z_out = math.sqrt(max(0.0, ro * ro - r_f * r_f))
    return (z_out - z_f) * (z_out / ro)


# Closest to flat (degrees) that any downward-facing part of the picture relief
# may be, and the slope of the short ramp that replaces the step between a
# picture edge and the frame.  At 0.12 mm layers a 20 degree surface moves
# sideways 0.33 mm per layer, so every new line still rests on the one below.
EDGE_RAMP_MIN_DEG = 20.0
# The same limit for the OUTSIDE of the wall when the two faces share an edge
# (imaging.outside_first).  The outside is the face you see, and whatever it
# does not carry the inside makes up, so it is held to the angle of the
# standard shoulder: a slope this printer is known to lay down cleanly
# (0.21 mm sideways per 0.12 mm layer, half a line).  Measured on the demo
# pictures, 30 instead of 20 degrees costs about a twentieth of the upper
# half's local contrast.
OUTSIDE_RAMP_MIN_DEG = 30.0
# Sharing an edge between the two faces must not leave the wall thinner than
# this share of the thinnest wall, measured square to the surface (see
# imaging.keep_true_thickness).  At 0.9 the thinnest wall of the default
# settings stays 0.72 mm: still two full lines of a 0.4 mm nozzle side by side.
TRUE_WALL_SHARE = 0.9
# With the relief on both sides ("split") it moves from the inside to the
# outside across this many degrees either side of the equator.
SPLIT_BLEND_DEG = 10.0
# Print-safe edges never flatten the picture below this slope (mm per mm).
MIN_RELIEF_SLOPE = 0.05


def split_share(phi: float) -> float:
    """"Both sides" relief: the part carried by the outside surface at latitude
    phi (radians) - none above the blend band, all of it below."""
    b = math.radians(SPLIT_BLEND_DEG)
    t = min(1.0, max(0.0, (b - phi) / (2 * b)))
    return t * t * (3.0 - 2.0 * t)


def relief_radius(p: Params) -> float:
    """Smallest radius a relief surface can have (the thickest wall, inside).
    Slopes are worked out at this radius, which errs on the safe side."""
    return max(5.0, p.diameter / 2 - max(p.t_max, p.t_frame))


def outside_first_radii(p: Params) -> tuple[float, float]:
    """Smallest radius the outside and the inside face of a picture can have
    when the relief is shared "outside first" (imaging.outside_first): the
    outside at the thinnest wall, the inside where it carries a whole edge."""
    ri = p.diameter / 2 - p.t_max                   # inside of the plain shell, see _frame_radii
    return max(5.0, ri + p.t_min), max(5.0, ri - (p.t_max - p.t_min))


def least_true_wall(p: Params) -> float:
    """Thinnest the wall may be anywhere, measured square to its surface."""
    return TRUE_WALL_SHARE * min(p.t_min, p.t_frame)


# The slicer lays two lines side by side - the outside and the inside wall loop -
# only where the wall is at least this many nozzle diameters wide (its narrowest
# line is 85 % of the nozzle).  Thinner places get a single line, and the two
# loops join there.  Sliced: with the 0.4 mm nozzle a thinnest wall of 0.8 mm
# keeps two loops on every layer, one of 0.6 mm loses them on 195 of 511.
TWO_LINES_NOZZLES = 1.7


def two_line_wall(p: Params) -> float:
    """Thinnest "thinnest wall" setting that still prints as two lines everywhere."""
    return TWO_LINES_NOZZLES * nozzle_for(p).diameter / TRUE_WALL_SHARE


def relief_reach(p: Params) -> float:
    """Largest difference between a picture's wall and the frame thickness."""
    return max(abs(p.t_max - p.t_frame), abs(p.t_min - p.t_frame))


def edge_ramp_slope(band_half: float) -> float:
    """Fastest the frame may thicken (mm per mm along the surface) next to the
    picture edge that prints as an overhang.  At that edge the wall already
    leans over at (90 - band_half) degrees from flat; thickening tips the
    overhanging surface further, and this slope stops at EDGE_RAMP_MIN_DEG.
    Further from the picture the wall leans more, so the ramp gets gentler."""
    return math.tan(math.radians(max(10.0, 90.0 - EDGE_RAMP_MIN_DEG - math.degrees(band_half))))


def frame_ramp_end(band_half: float, step: float, radius: float) -> float | None:
    """Latitude (radians from the equator) at which the frame ramp that starts
    at a picture edge has gained `step` mm, or None if it never gets there.

    The ramp thickens at tan(limit - latitude) mm per mm - always exactly
    EDGE_RAMP_MIN_DEG from flat - which integrates to
    radius * ln(cos(limit - latitude) / cos(limit - band_half))."""
    limit = math.radians(90.0 - EDGE_RAMP_MIN_DEG)
    x = math.cos(limit - band_half) * math.exp(step / radius)
    return None if x >= 1.0 else limit - math.acos(x)


def band_half_limit(p: Params) -> float:
    """Largest half-height of the picture band, in radians.

    The pictures must stay on the spherical part (clear of the shoulder and of
    the closed end), and the frame needs room to ramp up to its thickness
    between the picture edge and the point where the shell changes shape -
    otherwise the ramp would be cut short and leave a ledge."""
    phi_t = math.radians(90.0 - p.flare_angle)
    phi_c = math.radians(90.0 - CLOSURE_ANGLE_DEG)
    hard = min(phi_t, phi_c) - math.radians(3.0)
    if p.relief_side == "split":        # both picture edges face up on the printer: no ramp
        return hard
    if p.relief_side == "inside":       # the ramp runs towards the closed end
        step, limit = max(0.0, p.t_frame - p.t_min), phi_c
    else:                               # the ramp runs towards the cap
        step, limit = max(0.0, p.t_max - p.t_frame), phi_t
    radius = relief_radius(p)

    def fits(b: float) -> bool:
        end = frame_ramp_end(b, step, radius)
        return end is not None and end <= limit - math.radians(0.5)

    lo = math.radians(LIMITS["band_height_deg"][0]) / 2
    if fits(hard) or not fits(lo):
        return hard if fits(hard) else lo
    hi = hard
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def shoulder_for_cap(p: Params, cap_diameter: float) -> float | None:
    """Gentlest shoulder angle, in whole degrees and not below the present one,
    at which a cap this wide still fits under the sphere's shoulder - or None
    if no angle the app allows is steep enough.  (A steeper shoulder leaves the
    sphere higher up, where it is wider.)"""
    keep = p.flare_angle
    try:
        angle = math.ceil(keep - 1e-9)
        while angle <= LIMITS["flare_angle"][1] + 1e-9:
            p.flare_angle = float(angle)
            if cap_diameter_range(p)[1] >= cap_diameter - 1e-9:
                return max(keep, float(angle))
            angle += 1
        return None
    finally:
        p.flare_angle = keep


def cap_diameter_range(p: Params) -> tuple[float, float]:
    """Smallest / largest cap diameter that works with the other settings."""
    ro, ri, _ = _frame_radii(p)
    s = math.sin(math.radians(p.flare_angle))
    # the cap needs a wall round the widest thing cut into its top
    lo = (p.seat_diameter if p.seat else p.hole_diameter) + 2 * MIN_CAP_WALL
    hi = 2 * min(ro * s - 1.0, ri * s - 0.5 + p.cap_wall)
    return lo, hi


def derive(p: Params) -> Layout:
    """Work out the real dimensions; raise GeometryError if they cannot work."""
    notes: list[str] = []
    R = p.diameter / 2
    if p.t_max < p.t_min + 0.5:
        raise GeometryError("Thickest wall must be at least 0.5 mm more than the thinnest wall.",
                            "contrast")

    ro_frame, ri_frame, r_smooth = _frame_radii(p)

    theta_s = math.radians(p.flare_angle)
    phi_t = math.pi / 2 - theta_s
    rh = p.hole_diameter / 2
    rc = p.cap_diameter / 2
    r_seat = p.seat_diameter / 2 if p.seat else 0.0
    cap_lo, cap_hi = cap_diameter_range(p)
    if cap_hi < cap_lo:
        if p.seat:
            raise GeometryError(
                f"A {p.seat_diameter:.0f} mm seat for the light does not fit this sphere - use "
                f"a bigger sphere, a steeper shoulder angle or a smaller light.", "seat")
        raise GeometryError(
            f"A {p.hole_diameter:.0f} mm light opening does not fit this sphere - use a bigger "
            f"sphere, a steeper shoulder angle or a smaller opening.", "opening")
    if p.seat and p.hole_diameter > seat_opening_max(p) + 1e-9:
        raise GeometryError(
            f"The light opening must be at least {2 * SEAT_LEDGE_MIN:.0f} mm narrower than the "
            f"{p.seat_diameter:g} mm seat, so that the light has a ledge to rest on.", "window")
    if p.cap_diameter < cap_lo - 1e-9:
        what = (f"{p.seat_diameter:g} mm seat for the light" if p.seat
                else f"{p.hole_diameter:.0f} mm light opening")
        raise GeometryError(
            f"The cap must be at least {cap_lo:.1f} mm across to leave a wall around the "
            f"{what}.", "cap_small")
    if p.cap_diameter > cap_hi + 1e-9:
        raise GeometryError(
            f"The cap is too wide for this sphere - the largest that fits is {cap_hi:.0f} mm "
            f"(or use a steeper shoulder angle).", "cap_wide")

    # A cap noticeably wider than the opening is hollow (a cup with a top plate
    # around the opening); otherwise it is a plain tube.
    rci = rc - p.cap_wall
    cup = rci > rh + 0.4
    if not cup:
        rci = rh

    r_to, z_to = ro_frame * math.sin(theta_s), ro_frame * math.cos(theta_s)
    r_ti, z_ti = ri_frame * math.sin(theta_s), ri_frame * math.cos(theta_s)
    if rci > r_ti - 0.5 + 1e-9:
        raise GeometryError(
            f"A {p.hole_diameter:.0f} mm light opening does not fit this sphere - use a bigger "
            f"sphere, a thinner frame, a steeper shoulder angle or a smaller opening.", "opening")
    z_oc = z_to + (r_to - rc) * math.tan(theta_s)
    z_top = z_oc + p.cap_height
    # What the top of the cap is made of, from its top face down.  Plain: a plate
    # as thick as the cap wall (nothing at all for a tube).  With a seat: the
    # pocket, the cone the light rests on and a short straight neck.  z_plate is
    # where that ends - the ceiling of the hollow cap, seen from inside the ball.
    if p.seat:
        z_seat = z_top - p.seat_depth
        z_bore = z_seat - (r_seat - rh) * math.tan(math.radians(SEAT_CONE_DEG))
        z_plate = z_bore - SEAT_NECK_MM
    else:
        z_seat = z_bore = z_top
        z_plate = z_top - (p.cap_wall if cup else 0.0)
    z_ic = z_ti + (r_ti - rci) * math.tan(theta_s)
    if z_ic > z_plate - 0.5:
        need = z_ic + 0.5 + (z_top - z_plate) - z_oc
        if p.seat:
            raise GeometryError(f"The cap is too short for a seat {p.seat_depth:g} mm deep - make "
                                f"it at least {need:.1f} mm tall or the seat shallower.",
                                "cap_short")
        raise GeometryError(f"The cap is too short for its wall thickness - make it at least "
                            f"{need:.1f} mm tall or use a thinner cap wall.", "cap_short")

    phi_c, r_f, z_f = _closure(ro_frame, ri_frame)

    # Picture band: must stay on the spherical part, with room for the frame.
    max_half = band_half_limit(p)
    band_half = math.radians(p.band_height_deg) / 2
    if band_half > max_half + 1e-9:
        band_half = max_half
        notes.append(f"Picture height limited to {math.degrees(2 * band_half):.0f} deg so the "
                     f"pictures stay clear of the cap and of the closed end.")

    n = p.n_images
    sector = 2 * math.pi / n
    gap = math.radians(p.gap_deg)
    if gap > sector * 0.6:
        gap = sector * 0.6
        notes.append("Gap between pictures limited to 60% of each panel.")
    panel_half = (sector - gap) / 2
    centers = [(BACK_LON + (k + 0.5 + (n - 1) // 2) * sector) % (2 * math.pi) for k in range(n)]

    projection = p.projection
    if projection == "natural" and panel_half > math.radians(100):
        projection = "wrap"
        notes.append("A single picture wraps all the way round, so the Wrap projection is used.")

    height = z_top + ro_frame
    width = 2 * max(R, ro_frame)
    if p.relief_side == "split" and p.t_max > p.t_frame:
        # below the equator the darkest parts of the pictures stand proud of the frame;
        # the ball is widest where that has just begun
        blend = math.radians(SPLIT_BLEND_DEG)
        width = 2 * max((ro_frame + split_share(phi) * (p.t_max - p.t_frame)) * math.cos(phi)
                        for phi in (blend * (k / 40.0 - 1.0) for k in range(81)))
    if p.hanger != "none":
        width = max(width, 2 * hanger_dims(p, rc)["reach"])
    if height > MAX_PRINT_HEIGHT_MM - 2:
        raise GeometryError(f"The ornament would be {height:.0f} mm tall - too tall for the "
                            f"{PRINTER_NAME}.", "tall")
    return Layout(R=R, ro_frame=ro_frame, ri_frame=ri_frame, r_smooth=r_smooth, theta_s=theta_s,
                  phi_t=phi_t, rh=rh, rc=rc, rci=rci, cup=cup, z_plate=z_plate, seat=bool(p.seat),
                  r_seat=r_seat, z_seat=z_seat, z_bore=z_bore, r_to=r_to, z_to=z_to,
                  z_oc=z_oc, z_top=z_top, r_ti=r_ti, z_ti=z_ti, z_ic=z_ic, phi_c=phi_c,
                  r_f=r_f, z_f=z_f, roof=closure_roof_mm(ro_frame, r_f, z_f),
                  band_half=band_half,
                  panel_half=panel_half, sector=sector, centers=centers, projection=projection,
                  height=height, width=width, notes=notes)


def make_buildable(p: Params) -> bool:
    """Nudge linked settings, as little as possible, until derive() accepts them.

    The setting the user just changed is trusted as far as it can be; what gives
    way is, in order: the cap (to fit round the opening and under the shoulder),
    the cap height, then the opening itself.  Returns False only if no small
    repair exists (the caller then shows derive()'s message)."""
    for _ in range(300):
        try:
            derive(p)
            return True
        except GeometryError as exc:
            lo, hi = cap_diameter_range(p)
            if exc.kind == "contrast":
                p.t_max = p.t_min + 0.5
                if p.t_max > LIMITS["t_max"][1]:
                    p.t_max = LIMITS["t_max"][1]
                    p.t_min = p.t_max - 0.5
            elif exc.kind == "cap_small":
                p.cap_diameter = min(hi, math.ceil(lo * 10 - 1e-9) / 10)
            elif exc.kind == "cap_wide":
                p.cap_diameter = max(lo, math.floor(hi * 10 + 1e-9) / 10)
            elif exc.kind == "cap_short":
                if p.cap_height < LIMITS["cap_height"][1]:
                    p.cap_height = min(LIMITS["cap_height"][1], p.cap_height + 0.5)
                elif p.seat and p.seat_depth > LIMITS["seat_depth"][0]:
                    p.seat_depth = max(LIMITS["seat_depth"][0], p.seat_depth - 0.5)
                    p.light_preset = "custom"
                elif p.cap_wall > LIMITS["cap_wall"][0]:
                    p.cap_wall = max(LIMITS["cap_wall"][0], round(p.cap_wall - 0.1, 2))
                else:
                    return False
            elif exc.kind == "seat":
                # the seat is the widest thing in the cap: make it smaller, and give
                # it up altogether rather than leave a seat too small for any light
                p.light_preset = "custom"
                if p.seat_diameter > LIMITS["seat_diameter"][0]:
                    p.seat_diameter = max(LIMITS["seat_diameter"][0],
                                          math.ceil(p.seat_diameter * 2 - 1e-9) / 2 - 0.5)
                else:
                    p.seat = False
            elif exc.kind == "window":
                p.hole_diameter = max(LIMITS["hole_diameter"][0],
                                      math.floor(seat_opening_max(p) * 2 + 1e-9) / 2)
                p.light_preset = "custom"
                if p.hole_diameter > seat_opening_max(p) + 1e-9:
                    p.seat = False              # a seat too small to leave any ledge
            elif exc.kind == "opening":
                if p.hole_diameter <= LIMITS["hole_diameter"][0]:
                    return False
                p.hole_diameter = max(LIMITS["hole_diameter"][0],
                                      math.ceil(p.hole_diameter * 2 - 1e-9) / 2 - 0.5)
                p.light_preset = "custom"
            elif exc.kind == "tall":
                p.diameter = max(LIMITS["diameter"][0], p.diameter - 1.0)
            else:
                return False
    return False


def max_band_height_deg(p: Params) -> float:
    """Tallest picture band the other settings allow, in degrees."""
    return math.floor(2 * math.degrees(band_half_limit(p)) + 1e-6)


def nozzle_for(p: Params) -> Nozzle:
    """The nozzle these settings are printed with."""
    return NOZZLES.get(p.nozzle, NOZZLES[DEFAULT_NOZZLE])


def line_width(p: Params) -> float:
    """Width of one printed line, mm."""
    return nozzle_for(p).line_width


def mesh_pitch(p: Params) -> float:
    """Spacing of the samples of the exported model on the sphere's surface, mm:
    finer for a finer nozzle, which can reproduce finer detail."""
    pitches = nozzle_for(p).pitches
    return pitches.get(p.quality, pitches["standard"])


def recommended_layer_height(p: Params) -> float:
    """Tallest lithophane-grade layer whose sideways step on the shoulder stays
    under half an extrusion width (the usual limit for a clean overhang)."""
    nozzle = nozzle_for(p)
    limit = 0.5 * nozzle.line_width * math.tan(math.radians(p.flare_angle))
    for layer in nozzle.layers:
        if limit >= layer - 1e-9:
            return layer
    return nozzle.layers[-1]
