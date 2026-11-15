"""Photo loading, tone adjustment and projection of photos onto the sphere."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage

from .params import (EDGE_RAMP_MIN_DEG, MIN_RELIEF_SLOPE, OUTSIDE_RAMP_MIN_DEG, SPLIT_BLEND_DEG,
                     ImageSlot, Layout, Params, relief_radius, relief_reach)

MAX_SOURCE_PX = 4096          # longest side kept in memory
# Pillow refuses anything over ~179 megapixels as a possible "decompression bomb".
# These are the user's own photos and they are shrunk straight after loading, so
# allow up to 600 MP (200 MP phone photos exist).
Image.MAX_IMAGE_PIXELS = 600_000_000
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


@dataclass
class Photo:
    """A greyscale photo with a mip pyramid for alias-free sampling."""

    path: str
    mtime: float
    levels: list[np.ndarray]   # float32 in 0..1; levels[0] is full resolution
    lo: float                  # dark / bright points used by "auto levels"
    hi: float
    placeholder: bool = False
    plain: bool = False        # no photo at all: a plain panel for lettering and decorations

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.levels[0].shape
        return w, h


_cache: dict[str, Photo] = {}
_placeholders: dict[int, Photo] = {}


def _pyramid(gray: np.ndarray) -> list[np.ndarray]:
    levels = [np.ascontiguousarray(gray, dtype=np.float32)]
    while min(levels[-1].shape) >= 96:
        a = levels[-1]
        h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
        a = a[:h, :w]
        levels.append(np.ascontiguousarray(
            (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) * 0.25))
    return levels


def _photo_from_pil(img: Image.Image, path: str, mtime: float, placeholder: bool = False) -> Photo:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P", "PA") or "transparency" in img.info:
        # Flatten transparency onto white so see-through areas come out bright.
        rgba = img.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        flat.alpha_composite(rgba)
        img = flat
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        span = float(arr.max() - arr.min()) or 1.0
        img = Image.fromarray(((arr - arr.min()) / span * 255).astype(np.uint8))
    img = img.convert("L")
    if max(img.size) > MAX_SOURCE_PX:
        scale = MAX_SOURCE_PX / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.Resampling.LANCZOS)
    gray = np.asarray(img, dtype=np.float32) / 255.0
    lo, hi = np.percentile(gray[::2, ::2], (0.5, 99.5))
    if hi - lo < 0.05:        # flat image: leave the tones alone
        lo, hi = 0.0, 1.0
    return Photo(path=path, mtime=mtime, levels=_pyramid(gray), lo=float(lo), hi=float(hi),
                 placeholder=placeholder)


def load_photo(path: str) -> Photo:
    """Load (and cache) a photo. Raises OSError / ValueError if unreadable."""
    path = os.path.abspath(path)
    mtime = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit is not None and hit.mtime == mtime:
        return hit
    with Image.open(path) as img:
        img.load()
        photo = _photo_from_pil(img, path, mtime)
    _cache[path] = photo
    return photo


def placeholder_photo(index: int) -> Photo:
    """Stand-in shown in the preview for a slot that has no photo yet."""
    hit = _placeholders.get(index)
    if hit is not None:
        return hit
    size = 512
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    base = 0.55 + 0.25 * np.cos((xx - 0.5) * math.pi) * np.cos((yy - 0.5) * math.pi)
    stripes = 0.06 * np.sign(np.sin((xx + yy) * 40.0))
    img = Image.fromarray(np.clip((base + stripes) * 255, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        big = ImageFont.load_default(size=230)
        small = ImageFont.load_default(size=40)
    except TypeError:          # very old Pillow
        big = small = ImageFont.load_default()
    draw.text((size / 2, size / 2 - 30), str(index + 1), fill=20, font=big, anchor="mm")
    draw.text((size / 2, size - 70), "add a photo", fill=20, font=small, anchor="mm")
    photo = _photo_from_pil(img, f"<placeholder {index + 1}>", 0.0, placeholder=True)
    photo.lo, photo.hi = 0.0, 1.0
    _placeholders[index] = photo
    return photo


_PLAIN = Photo(path="", mtime=0.0, levels=[np.zeros((2, 2), dtype=np.float32)], lo=0.0, hi=1.0,
               plain=True)


def photos_for(p: Params) -> tuple[list[Photo], list[str]]:
    """Photos for the active slots (placeholders where missing) plus any problems.

    A slot without a photo that carries lettering or a decoration is meant to
    be that way - a picture that is only text - and gets a plain panel, not a
    placeholder and not a complaint."""
    photos: list[Photo] = []
    problems: list[str] = []
    for k in range(p.n_images):
        slot = p.images[k]
        if slot.path:
            try:
                photos.append(load_photo(slot.path))
                continue
            except Exception as exc:      # unreadable / missing file
                problems.append(f"Picture {k + 1}: cannot read '{os.path.basename(slot.path)}' "
                                f"({exc.__class__.__name__}).")
        elif slot.decorated:
            photos.append(_PLAIN)
            continue
        else:
            problems.append(f"Picture {k + 1} has no photo yet.")
        photos.append(placeholder_photo(k))
    return photos, problems


def oriented(level: np.ndarray, slot: ImageSlot) -> np.ndarray:
    """Apply the slot's rotation / mirror to a pyramid level (cheap views)."""
    out = np.rot90(level, k=-(slot.rotation // 90) % 4)
    if slot.flip:
        out = out[:, ::-1]
    return out


def thumbnail(photo: Photo, slot: ImageSlot, size: int = 96) -> np.ndarray:
    """Small uint8 greyscale preview (h, w) honouring rotation / mirror / tone."""
    level = photo.levels[-1]
    for cand in reversed(photo.levels):
        level = cand
        if max(cand.shape) >= size:
            break
    arr = apply_tone(oriented(level, slot), photo, slot)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(img))


def apply_tone(values: np.ndarray, photo: Photo, slot: ImageSlot) -> np.ndarray:
    v = values.astype(np.float32, copy=True)
    if slot.auto_levels:
        v = (v - photo.lo) / max(1e-3, photo.hi - photo.lo)
    v = (v - 0.5) * slot.contrast + 0.5 + slot.brightness
    return np.clip(v, 0.0, 1.0, out=v)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def panel_uv(dlam: np.ndarray, phi: np.ndarray, projection: str) -> tuple[np.ndarray, np.ndarray]:
    """Map a point of a panel (longitude offset, latitude) to flat picture
    coordinates in radians of arc (U to the right, V up).

    "wrap"    - plain longitude / latitude: the whole photo fits the panel, but
                it gets narrower towards the top and bottom of the sphere.
    "natural" - azimuthal-equidistant about the panel centre: distances from the
                centre are true, so faces keep their proportions; the panel then
                shows a slightly barrel-shaped part of the photo.
    """
    if projection == "wrap":
        return dlam, phi
    cos_c = np.clip(np.cos(phi) * np.cos(dlam), -1.0, 1.0)
    c = np.arccos(cos_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(c > 1e-6, c / np.sin(c), 1.0)
    return k * np.cos(phi) * np.sin(dlam), k * np.sin(phi)


def window_extent(panel_half: float, band_half: float, projection: str) -> tuple[float, float]:
    """Half-size (U, V) of the box the photo must cover for a panel."""
    edge_l = np.linspace(-panel_half, panel_half, 65)
    edge_p = np.linspace(-band_half, band_half, 65)
    dl = np.concatenate([edge_l, edge_l, np.full(65, -panel_half), np.full(65, panel_half)])
    ph = np.concatenate([np.full(65, -band_half), np.full(65, band_half), edge_p, edge_p])
    u, v = panel_uv(dl, ph, projection)
    return float(np.abs(u).max()), float(np.abs(v).max())


def sample_photo(photo: Photo, slot: ImageSlot, U: np.ndarray, V: np.ndarray,
                 u_max: float, v_max: float, pitch_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample a photo at flat picture coordinates.

    Returns (values in 0..1 after tone adjustments, mask of points inside the photo).
    """
    w, h = oriented(photo.levels[0], slot).shape[::-1]
    scale = max(2 * u_max / w, 2 * v_max / h) * slot.zoom      # radians per source pixel
    ratio = pitch_rad / scale                                   # source pixels per mesh sample
    level = 0
    if ratio > 1.0:
        level = min(len(photo.levels) - 1, int(math.floor(math.log2(ratio) + 0.35)))
    img = oriented(photo.levels[level], slot)
    hl, wl = img.shape
    sx, sy = scale * w / wl, scale * h / hl
    slack_x = (w * scale - 2 * u_max) / 2
    slack_y = (h * scale - 2 * v_max) / 2
    ox, oy = slot.shift_x * abs(slack_x), slot.shift_y * abs(slack_y)
    px = (wl - 1) / 2 + (U - ox) / sx
    py = (hl - 1) / 2 - (V - oy) / sy
    eps = 1e-6      # a point exactly on the photo's border belongs to it despite rounding
    mask = (px >= -0.5 - eps) & (px <= wl - 0.5 + eps) & (py >= -0.5 - eps) & (py <= hl - 0.5 + eps)
    values = ndimage.map_coordinates(np.ascontiguousarray(img), [py, px], order=1, mode="nearest")
    return apply_tone(values, photo, slot), mask


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Which surface carries the picture, and what the printer can build there
# ---------------------------------------------------------------------------
# The ornament is printed upside-down (cap on the bed), so "up" on the printer
# is towards LOWER latitude.  Seen from the printer:
#
#   * above the equator (the half printed first) the wall leans outwards as it
#     rises: its inside faces up like a floor, its outside faces down;
#   * below the equator (the half printed last) the wall leans inwards: its
#     outside faces up, its inside is a ceiling.
#
# Relief on a face that points up can have any shape.  Relief on a face that
# points down sticks out of a leaning ceiling: where the wall thickens going
# up, the new plastic reaches out - and, if it thickens too fast, DOWN - into
# empty air.  A bump whose tip is lower than the place where it joins the wall
# starts in mid-air; Bambu Studio reports those as "floating regions".
def relief_share(side: str, phi: np.ndarray) -> np.ndarray:
    """Part of the picture relief carried by the OUTSIDE surface at latitude
    phi (radians): 0 = all of it inside, 1 = all of it outside.

    "split" puts the relief on whichever face points up on the printer -
    inside near the cap, outside near the closed end - and cross-fades over a
    narrow band at the equator, where the wall is upright and either face
    prints equally well."""
    phi = np.asarray(phi, dtype=np.float64)
    if side == "inside":
        return np.zeros_like(phi)
    if side == "outside":
        return np.ones_like(phi)
    if side != "split":
        raise ValueError(f"unknown relief side {side!r}")
    b = math.radians(SPLIT_BLEND_DEG)
    t = np.clip((b - phi) / (2 * b), 0.0, 1.0)         # same curve as params.split_share
    return t * t * (3.0 - 2.0 * t)


def _tan_below_90(angle: np.ndarray) -> np.ndarray:
    """tan(angle), 0 for negative angles and infinite from 89 degrees on."""
    out = np.full(angle.shape, np.inf)
    ok = angle < math.radians(89.0)
    out[ok] = np.tan(np.maximum(angle[ok], 0.0))
    return out


def overhang_slope_limit(side: str, phis: np.ndarray, radius: float, reach: float = 0.0) -> np.ndarray:
    """Fastest the wall may thicken going up on the printer, in mm per mm of
    surface, between each pair of neighbouring rows (phis ascending), so that
    no downward-facing part of the relief is closer to flat than
    EDGE_RAMP_MIN_DEG.  np.inf where nothing limits it.

    A face that already leans over by mu degrees and thickens at s mm/mm
    overhangs by mu + atan(s); keeping that under the limit gives
    s <= tan(limit - mu).  For the face that leans the other way (it points
    up) mu is negative, so a few degrees from the equator nothing limits it.

    reach is the largest |thickness - frame thickness| in the picture; "split"
    needs it because moving the relief from one face to the other across the
    equator tilts both faces a little, and that uses up some of the allowance.
    """
    mid = 0.5 * (phis[:-1] + phis[1:])
    limit = math.radians(90.0 - EDGE_RAMP_MIN_DEG)
    g_in = _tan_below_90(limit + mid)          # inside face: a ceiling below the equator
    g_out = _tan_below_90(limit - mid)         # outside face: an underside above the equator
    if side == "inside":
        slope = g_in
    elif side == "outside":
        slope = g_out
    else:
        share = relief_share(side, phis)
        a = 0.5 * (share[:-1] + share[1:])
        tilt = reach * np.abs(np.diff(share)) / (np.diff(phis) * radius)
        with np.errstate(divide="ignore", invalid="ignore"):
            lim_out = np.where(a > 1e-9, (g_out - tilt) / np.maximum(a, 1e-9), np.inf)
            lim_in = np.where(a < 1 - 1e-9, (g_in - tilt) / np.maximum(1 - a, 1e-9), np.inf)
        slope = np.minimum(lim_out, lim_in)
    return np.maximum(slope, MIN_RELIEF_SLOPE)


def limit_overhang_slope(T: np.ndarray, phis: np.ndarray, radius: float, side: str,
                         reach: float = 0.0) -> np.ndarray:
    """Soften the edges the printer cannot build ("print-safe edges").

    Wherever the wall thickens going up on the printer faster than
    overhang_slope_limit() allows, the edge is replaced by the steepest ramp
    that still prints, centred on the edge.  Everything else is returned
    unchanged.  After this no part of the relief hangs lower than the place
    where it joins the wall, so nothing starts in mid-air.

    Which edges that touches depends on the side that carries the relief:
    with "split" only extreme edges right at the equator (hardly ever); with
    "inside" the bright-above-dark edges in the lower half of the pictures,
    more the nearer the closed end; with "outside" the same edges in the upper
    half.  T has one row per latitude in phis (ascending); radius is the
    smallest radius the relief surface can have.
    """
    if len(phis) < 2:
        return T
    step = overhang_slope_limit(side, phis, radius, reach) * np.diff(phis) * radius
    thick = T.copy()            # version A: start thickening earlier (below, on the printer)
    for j in range(len(phis) - 1):
        np.maximum(thick[j + 1], thick[j] - step[j], out=thick[j + 1])
    thin = T.copy()             # version B: finish thickening later (above, on the printer)
    for j in range(len(phis) - 2, -1, -1):
        np.minimum(thin[j], thin[j + 1] + step[j], out=thin[j])
    return 0.5 * (thick + thin)   # both obey the limit, so does their mean; the edge stays put


def outside_first(extra: np.ndarray, phis: np.ndarray, r_outside: float, r_inside: float,
                  wall: float | None = None, lon_step: float = 0.0,
                  least: float = 0.0, inside_extra: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Share the picture relief between the two faces of the wall so that the
    OUTSIDE carries as much of it as the printer can build there.

    Why the outside: lit from within, relief on the inside of the wall looks
    blurred.  The light is shaped by the relief first and then spreads
    sideways in the plastic before it leaves through a smooth outside, and
    pits on the lit face are partly in shadow.  Relief on the outside is the
    last thing the light meets, so it stays sharp.

    Why not all of it: in the half printed first the outside faces down.  It
    can step inwards going up on the printer as suddenly as it likes (that
    face points up), but where the wall THICKENS going up the outside may
    only lean out to OUTSIDE_RAMP_MIN_DEG from flat.  So the outside surface is
    the buildable surface closest to the picture (least squares): identical
    to it wherever the picture can be built, and a ramp of the steepest
    buildable slope, centred on the edge, where an edge is too sudden.  What
    the wall then lacks or has too much of is made up on the inside, which
    faces up in that half and can take a step - so the wall is still exactly
    as thick as the picture asks.  Within about 20 degrees of the equator the
    inside is limited as well (to EDGE_RAMP_MIN_DEG, like every hidden face);
    an extreme edge there is spread over a fraction of a millimetre.

    One more thing limits the inside.  Where it makes room for an outside that
    is still on its ramp, both faces move outwards together, and a wall that
    is bodily displaced - suddenly between two neighbouring columns, or
    steadily along a steep ramp - is thinner across than it is along the
    radius.  With wall (the frame thickness), lon_step (radians between
    columns) and least (the thinnest the wall may be, measured square to its
    surface) given, the inside gives way there and the wall comes out a little
    thicker than the picture asks instead (keep_true_thickness).

    Of the ways of sharing an edge that were simulated (light scattering in
    the wall, see the README), this one kept small bright dots and thin dark
    lines closest to a relief that is entirely outside, and it is the least
    sensitive to light that skims along the wall.

    extra: thickness wanted beyond the frame thickness (rows = latitudes
    phis, ascending, including plain rows either side of the pictures).
    r_outside / r_inside: smallest radius either face can have (slopes are
    worked out there, which errs on the safe side).  Returns (how far the
    outside moves out, how far the inside moves in).

    inside_extra: more thickness wanted, which is to go on the inside whatever
    the outside could carry.  That is for the strip between the pictures and
    the shoulder: the outside leans over so far there that it could take only
    a sliver of an edge, and only by starting its ramp centimetres lower -
    across the top of the pictures, which it would blur.
    """
    extra = np.asarray(extra, dtype=np.float64)
    if len(phis) < 2:
        return extra.copy(), np.zeros_like(extra)
    mid = 0.5 * (phis[:-1] + phis[1:])
    arc = np.diff(phis)
    # outside: an underside above the equator; inside: a ceiling below it
    step_out = _tan_below_90(math.radians(90.0 - OUTSIDE_RAMP_MIN_DEG) - mid) * arc * r_outside
    step_in = _tan_below_90(math.radians(90.0 - EDGE_RAMP_MIN_DEG) + mid) * arc * r_inside
    weights = np.gradient(phis)                     # each row stands for this much of the meridian
    weights = weights / weights.max()
    out = closest_buildable(extra, step_out, weights)
    rest = extra - out
    if inside_extra is not None:
        rest = rest + inside_extra
    inside = closest_buildable(rest, step_in, weights)
    if wall is not None and least > 0.0 and lon_step > 0.0:
        guarded = keep_true_thickness(wall + out, inside, phis, lon_step, r_inside, least)
        if guarded is not inside:
            # filling a dip may leave an edge the inside cannot be printed over where it
            # is a ceiling: start that thickening earlier (only ever adds to the wall)
            inside = guarded
            for j in np.nonzero(step_in < _NO_LIMIT_MM)[0]:
                np.maximum(inside[j + 1], inside[j] - step_in[j], out=inside[j + 1])
    return out, inside


def keep_true_thickness(outer: np.ndarray, inside: np.ndarray, phis: np.ndarray, lon_step: float,
                        radius: float, least: float) -> np.ndarray:
    """Fill the dips of the inside surface wherever they would leave the wall
    thinner than `least` mm measured square to its surface - in space, not
    along the radius.

    outer: height of the outside surface above the plain inside sphere (rows =
    latitudes phis, columns lon_step radians apart).  inside: how far the
    inside surface stands in from that sphere; negative where it has made
    room by moving outwards.  radius: of that sphere.

    A point of the inside surface is at least `least` away from the outside
    surface exactly when a ball of that radius centred on it stays under the
    outside surface.  So the inside may rise no higher than the outside
    surface lowered by such a ball (a grey-scale erosion), and where a dip
    reaches higher it is filled up to there.  Returns `inside` itself if
    nothing had to change."""
    dips = (inside < -1e-9).any(axis=1)
    if not dips.any():
        return inside
    n_rows, n_lon = inside.shape
    arc = np.asarray(phis, dtype=np.float64) * radius           # each row's place on the meridian
    col = np.cos(phis) * radius * lon_step                      # spacing of the columns in each row
    rows = np.nonzero(dips)[0]
    j0, j1 = int(rows[0]), int(rows[-1]) + 1
    target = np.arange(j0, j1)
    # how many rows / columns away a point within `least` mm can be (where the rows are closest)
    closest_rows = float(np.diff(arc[max(j0 - 1, 0):min(j1 + 1, n_rows)]).min()) if n_rows > 1 else least
    reach_rows = int(math.ceil(least / max(1e-6, closest_rows)))
    reach_cols = int(math.ceil(least / max(1e-6, float(col[j0:j1].min()))))
    ceiling = np.full((j1 - j0, n_lon), np.inf)
    for dj in range(-reach_rows, reach_rows + 1):
        source = np.clip(target + dj, 0, n_rows - 1)
        along = arc[source] - arc[target]
        strip = outer[source]
        for dk in range(-reach_cols, reach_cols + 1):
            room = least * least - along * along - (dk * col[target]) ** 2
            near = room >= 0.0
            if not near.any():
                continue
            ball = np.where(near, np.sqrt(np.maximum(room, 0.0)), -np.inf)
            np.minimum(ceiling, (np.roll(strip, -dk, axis=1) if dk else strip) - ball[:, None],
                       out=ceiling)
    lifted = np.maximum(inside[j0:j1], -ceiling)
    if np.array_equal(lifted, inside[j0:j1]):
        return inside
    out = inside.copy()
    out[j0:j1] = lifted
    return out


_NO_LIMIT_MM = 100.0        # stands in for "no limit" between two rows (walls are under 4 mm)


def closest_buildable(surface: np.ndarray, step: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The surface closest to `surface` (weighted least squares, column by
    column) that never rises by more than step[j] from row j + 1 to row j -
    that is, going up on the printer.  Columns that already obey the limit
    come back untouched; the first and last rows are never moved (they join
    the plain shell).

    Subtracting a ramp that climbs by exactly the allowed step turns "rises
    no faster than" into "never rises", which is an isotonic regression."""
    from scipy.optimize import isotonic_regression

    step = np.minimum(np.asarray(step, dtype=np.float64), _NO_LIMIT_MM)
    too_steep = ((surface[:-1] - surface[1:]) > step[:, None] + 1e-12).any(axis=0)
    out = surface.copy()
    if not too_steep.any():
        return out
    ramp = np.concatenate([np.cumsum(step[::-1])[::-1], [0.0]])     # ramp[j] - ramp[j + 1] = step[j]
    w = np.asarray(weights, dtype=np.float64).copy()
    w[0] = w[-1] = 1.0e9
    for c in np.nonzero(too_steep)[0]:
        fitted = isotonic_regression(surface[:, c] - ramp, weights=w, increasing=True).x + ramp
        same = np.abs(fitted - surface[:, c]) < 1e-9        # rounding only: keep the original
        out[:, c] = np.where(same, surface[:, c], fitted)
    return out


def thickness_band(p: Params, lay: Layout, lam: np.ndarray, phis: np.ndarray,
                   photos: list[Photo], pitch_mm: float) -> np.ndarray:
    """Wall thickness (mm) on the grid phis x lam covering the picture band.

    Points that belong to no picture (dividers, uncovered margins) get the
    frame thickness.  With the relief on the outside this is the thickness the
    picture asks for; what the printer can build is settled when the relief is
    shared between the two faces (outside_first).  With the other settings
    print-safe edges are applied here.
    """
    from . import overlay           # (it draws with this module's projections)

    T = np.full((len(phis), len(lam)), p.t_frame, dtype=np.float64)
    u_max, v_max = window_extent(lay.panel_half, lay.band_half, lay.projection)
    pitch_rad = pitch_mm / lay.R
    full_wrap = lay.n_images == 1 and lay.panel_half >= math.pi - 1e-9
    panel = overlay.Panel.of(lay)
    for k, center in enumerate(lay.centers):
        dlam_all = wrap_angle(lam - center)
        cols = np.nonzero(np.abs(dlam_all) <= lay.panel_half + 1e-12)[0]
        if cols.size == 0:
            continue
        cols = cols[np.argsort(dlam_all[cols])]
        dl, ph = np.meshgrid(dlam_all[cols], phis)
        slot = p.images[k]
        if photos[k].plain:             # no photo: a plain panel for the lettering
            t = np.full(dl.shape, overlay.backdrop_thickness(slot, p), dtype=np.float64)
        else:
            U, V = panel_uv(dl, ph, lay.projection)
            values, mask = sample_photo(photos[k], slot, U, V, u_max, v_max, pitch_rad)
            if p.smoothing > 0.01:
                mode = ("nearest", "wrap") if full_wrap else "nearest"
                values = ndimage.gaussian_filter(values, p.smoothing, mode=mode)
            if abs(p.gamma - 1.0) > 1e-3:
                values = np.power(values, 1.0 / p.gamma)
            if p.invert:
                values = 1.0 - values
            t = p.t_max - (p.t_max - p.t_min) * values.astype(np.float64)
            t[~mask] = p.t_frame
        # lettering and decorations go on last, in the tones of the finished wall:
        # "glowing" stays the thinnest wall whatever was done to the photo's tones
        layer = overlay.layer_for(slot, panel, pitch_mm)
        if layer is not None:
            t = overlay.apply(t, layer, dl, ph, p)
        T[:, cols] = t
    if p.safe_edges and p.relief_side != "outside":
        T = limit_overhang_slope(T, phis, relief_radius(p), p.relief_side, relief_reach(p))
    return T
