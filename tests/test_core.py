"""Tests for the geometry engine: does every reasonable setting give a valid,
printable solid with the picture the right way round?

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithosphere import checks, export, geometry, imaging, render  # noqa: E402
from lithosphere.params import (CLOSURE_ANGLE_DEG, CLOSURE_ROOF_MM, EDGE_RAMP_MIN_DEG,  # noqa: E402
                                OUTSIDE_RAMP_MIN_DEG,
                                LIGHT_PRESETS, LIMITS, RELIEF_SIDES, SPLIT_BLEND_DEG, GeometryError,
                                ImageSlot,
                                MIN_CAP_WALL, NOZZLES, QUALITY_PRESETS, SEAT_CLEARANCE_MM,
                                SEAT_CONE_DEG, SEAT_LEDGE_MIN, TRUE_WALL_SHARE, least_true_wall,
                                Params, band_half_limit, cap_diameter_range, closure_profile,
                                derive, edge_ramp_slope, frame_ramp_end, line_width, make_buildable,
                                max_band_height_deg, mesh_pitch, outside_first_radii,
                                params_from_file, recommended_layer_height, relief_radius,
                                seat_opening_max, shoulder_for_cap, split_share)

PITCH = 0.6          # coarse mesh keeps the tests quick
SIDES = ["split", "inside", "outside"]


def make_image(tmp_path, name: str, array: np.ndarray) -> str:
    path = str(tmp_path / name)
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


@pytest.fixture()
def gradient(tmp_path) -> str:
    """Left-to-right ramp, dark to bright, slightly brighter towards the top."""
    x = np.linspace(0, 235, 300)[None, :]
    y = np.linspace(20, 0, 240)[:, None]
    return make_image(tmp_path, "gradient.png", np.clip(x + y, 0, 255))


def with_photos(p: Params, path: str) -> Params:
    for slot in p.images:
        slot.path = path
    return p


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
def test_params_round_trip():
    p = Params(diameter=95, n_images=2, relief_side="outside", hanger="one")
    p.images[1] = ImageSlot(path="x.png", zoom=1.7, rotation=90, flip=True)
    q = Params.from_json(p.to_json())
    assert q.to_dict() == p.to_dict()


def test_sanitize_clamps_bad_values():
    p = Params.from_dict({"diameter": 9999, "n_images": 99, "projection": "nonsense",
                          "t_min": float("nan"), "images": [{"zoom": -5, "rotation": 450}],
                          "unknown_key": 1})
    assert p.diameter == 150 and p.n_images == 4 and p.projection == "natural"
    assert p.t_min == 0.4 and p.images[0].zoom == 0.5 and p.images[0].rotation == 90
    assert len(p.images) == 4


def test_derive_rejects_impossible_shapes():
    with pytest.raises(GeometryError):
        derive(Params(t_min=1.5, t_max=1.6))                     # no contrast range
    with pytest.raises(GeometryError):
        derive(Params(diameter=50, hole_diameter=40, cap_diameter=45))   # opening too big
    with pytest.raises(GeometryError):
        derive(Params(hole_diameter=20, cap_diameter=21))        # no wall around the opening
    with pytest.raises(GeometryError):
        derive(Params(cap_diameter=58))                          # cap wider than the shoulder


def test_defaults_are_valid_and_pass_every_check(gradient):
    p = with_photos(Params(), gradient)
    lay = derive(p)
    assert not lay.notes
    results = checks.analyze(p, lay, 33.0, [])
    assert checks.worst_level(results) == checks.OK, [c for c in results if c.level != checks.OK]
    assert recommended_layer_height(p) == 0.12


def test_every_light_preset_fits_after_cap_fit():
    for key, preset in LIGHT_PRESETS.items():
        if preset.hole is None:
            continue
        p = Params(light_preset=key, hole_diameter=preset.hole)
        if preset.seat is not None:
            p.seat, p.seat_diameter, p.seat_depth = True, preset.seat, preset.depth
            # a wide light may need a steeper shoulder, which the app then chooses
            angle = shoulder_for_cap(p, preset.seat + 2 * MIN_CAP_WALL + 0.1)
            assert angle is not None and angle <= 36.0, key
            p.flare_angle = angle
        lo, hi = cap_diameter_range(p)
        assert hi >= lo, key
        p.cap_diameter = min(hi, max(lo, p.cap_diameter))
        assert make_buildable(p), key               # at most the cap grows taller for a seat
        derive(p)
        assert p.hole_diameter == preset.hole and p.light_preset == key
        assert p.seat == (preset.seat is not None)
        if p.seat:
            assert (p.seat_diameter, p.seat_depth) == (preset.seat, preset.depth)


@pytest.mark.parametrize("ro,ri", [(40, 37.8), (25, 22.8), (75, 72.8), (40, 39.2), (40, 36.0),
                                   (25, 21.0)])
def test_closure_roof_is_thick_enough(ro, ri):
    theta_c, r_c, r_f, z_f = closure_profile(ro, ri)
    assert math.isclose(math.degrees(theta_c), CLOSURE_ANGLE_DEG)
    assert 0 <= r_f <= r_c
    # roof above the rim of the flat ceiling, and at the pole
    rim_roof = math.sqrt(ro * ro - r_f * r_f) - z_f
    assert rim_roof >= min(CLOSURE_ROOF_MM, 0.6) - 1e-6
    assert ro - z_f >= rim_roof - 1e-9
    assert z_f < ro


# ---------------------------------------------------------------------------
# picture placement
# ---------------------------------------------------------------------------
def test_projections_agree_at_the_panel_centre():
    dl = np.array([0.0, 1e-4, 0.0])
    ph = np.array([0.0, 0.0, 1e-4])
    for proj in ("natural", "wrap"):
        u, v = imaging.panel_uv(dl, ph, proj)
        assert np.allclose(u, dl, atol=1e-9) and np.allclose(v, ph, atol=1e-9)


def test_natural_projection_keeps_distances_from_the_centre():
    dl, ph = np.meshgrid(np.linspace(-1.0, 1.0, 21), np.linspace(-0.9, 0.9, 19))
    u, v = imaging.panel_uv(dl, ph, "natural")
    arc = np.arccos(np.cos(ph) * np.cos(dl))          # great-circle distance from the centre
    assert np.allclose(np.hypot(u, v), arc, atol=1e-9)


def _thickness_grid(p: Params, pitch: float = 0.5):
    lay = derive(p)
    photos, problems = imaging.photos_for(p)
    assert not problems
    n_lon = geometry.longitude_count(lay.R, pitch)
    lam = np.arange(n_lon) * (2 * math.pi / n_lon)
    phis, band = geometry._relief_rows(lay, pitch, -math.pi / 2)
    T = imaging.thickness_band(p, lay, lam, phis[band], photos, pitch)
    return lay, lam, phis[band], T


@pytest.mark.parametrize("projection", ["natural", "wrap"])
def test_picture_is_not_mirrored_or_upside_down(gradient, projection):
    """The gradient is bright on its right, so the wall must be thinner at larger
    longitude (to the right for a viewer outside) and slightly thinner higher up."""
    p = with_photos(Params(n_images=3, projection=projection, smoothing=0.0), gradient)
    lay, lam, phis, T = _thickness_grid(p)
    center = lay.centers[0]
    d = imaging.wrap_angle(lam - center)
    mid = len(phis) // 2
    left = T[mid, np.argmin(np.abs(d + 0.6 * lay.panel_half))]
    right = T[mid, np.argmin(np.abs(d - 0.6 * lay.panel_half))]
    assert right < left - 0.5, (left, right)
    col = np.argmin(np.abs(d))
    assert T[-3, col] < T[2, col]                      # top of the photo is the brighter end


def test_thickness_range_and_dividers(gradient):
    p = with_photos(Params(n_images=3, gap_deg=8.0), gradient)
    lay, lam, _phis, T = _thickness_grid(p)
    assert T.min() >= p.t_min - 1e-6 and T.max() <= p.t_max + 1e-6
    # auto levels spreads the photo over the range (the crop trims the ramp's ends a little)
    assert T.max() - T.min() > 0.7 * (p.t_max - p.t_min)
    for k in range(3):                                          # divider columns are plain frame
        divider = lay.centers[k] + lay.sector / 2
        col = np.argmin(np.abs(imaging.wrap_angle(lam - divider)))
        assert np.allclose(T[:, col], p.t_frame)
    back = np.argmin(np.abs(imaging.wrap_angle(lam - math.pi / 2)))
    assert np.allclose(T[:, back], p.t_frame), "a divider must face the back for the seam"


def test_invert_and_rotation(gradient):
    p = with_photos(Params(n_images=2, smoothing=0.0, safe_edges=False), gradient)
    _lay, _lam, _phis, normal = _thickness_grid(p)
    p.invert = True
    _lay, _lam, _phis, negative = _thickness_grid(p)
    in_picture = ~np.isclose(normal, p.t_frame)
    assert np.allclose((normal + negative)[in_picture], p.t_min + p.t_max, atol=1e-6)
    p.invert = False
    for slot in p.images:
        slot.rotation = 180
    _lay, _lam, _phis, turned = _thickness_grid(p)
    assert not np.allclose(turned, normal)


def _steepest_overhang(T, phis, side, ro=40.0, t_frame=2.2):
    """Angle from flat (degrees) of the most overhanging piece of the two surfaces
    that a thickness map gives, measured up each meridian as it is printed (cap
    down, so up on the printer is towards lower latitude).  90 = upright,
    0 = a flat ledge, negative = the surface comes back DOWN: a tip in mid-air."""
    share = imaging.relief_share(side, phis)[:, None]
    extra = T - t_frame
    return _steepest_of(ro + share * extra, ro - t_frame - (1 - share) * extra, phis)


def _steepest_face(r, phis, towards_air):
    """The same for one given surface (radius per latitude row and column); towards_air is
    +1 for the outside of the wall, -1 for the inside."""
    cp, sp = np.cos(phis)[:, None], np.sin(phis)[:, None]
    rho, z = r * cp, -r * sp
    out = (rho[:-1] - rho[1:]) * towards_air           # the surface moves out over its air side
    up = z[:-1] - z[1:]
    hang = out > 1e-12
    return float(np.degrees(np.arctan2(up[hang], out[hang])).min()) if hang.any() else 90.0


def _steepest_of(r_outside, r_inside, phis):
    """... and for the two surfaces of a wall together."""
    return min(_steepest_face(r_outside, phis, 1.0), _steepest_face(r_inside, phis, -1.0))


@pytest.mark.parametrize("side", SIDES)
def test_print_safe_edges_leave_nothing_hanging(side):
    """Random 'worst photo': unlimited it is full of flat ledges and tips that hang in
    mid-air; limited, no downward-facing part of either surface is closer to flat
    than the design allows - and a normal picture is barely changed."""
    rng = np.random.default_rng(1)
    t_frame, radius, reach = 2.2, 40.0 - 2.8, 1.4
    phis = np.radians(np.linspace(-50, 50, 281))
    T = rng.choice([0.8, 2.8], size=(len(phis), 60)).astype(float)
    assert _steepest_overhang(T, phis, side) < EDGE_RAMP_MIN_DEG - 3
    out = imaging.limit_overhang_slope(T, phis, radius, side, reach)
    assert _steepest_overhang(out, phis, side) >= EDGE_RAMP_MIN_DEG - 0.3
    assert out.min() >= 0.8 - 1e-9 and out.max() <= 2.8 + 1e-9
    # a smooth picture is not touched at all
    smooth = 1.8 + 0.9 * np.sin(np.linspace(0, 3, len(phis)))[:, None] * np.ones((1, 5))
    assert np.allclose(imaging.limit_overhang_slope(smooth, phis, radius, side, reach), smooth)
    # a single sharp edge stays centred: as much is added below it as is removed above it
    edge = np.full((len(phis), 1), 0.8)
    edge[:140] = 2.8                            # dark below (lower latitude), bright above
    limited = imaging.limit_overhang_slope(edge, phis, radius, side, reach)
    changed = np.abs(limited - edge).sum()
    assert abs((limited - edge).sum()) < 0.05 * changed
    assert 0 < changed * np.diff(phis)[0] * 40.0 < 2.0     # mm^2: a small ramp


def test_which_edges_print_safe_edges_touch():
    """A crisp everyday edge (2 mm of wall gained over 1 mm, going up on the printer),
    repeated at every latitude.  On a face that points up it prints as it is; on a
    leaning ceiling it has to be eased - which is why "split" exists."""
    phis = np.radians(np.linspace(-50, 50, 401))
    arc = -phis * 40.0                                   # mm travelled up the printer
    T = (0.8 + 2.0 * np.clip((arc % 3.0), 0.0, 1.0))[:, None] * np.ones((1, 3))
    lat = np.degrees(phis)
    radius, reach = 40.0 - 2.8, 1.4
    changed = {side: np.abs(imaging.limit_overhang_slope(T, phis, radius, side, reach) - T).max(axis=1)
               for side in SIDES}
    assert changed["split"].max() < 1e-9                 # untouched everywhere
    assert changed["inside"][lat > 0].max() < 1e-9       # the inside is a floor above the equator
    assert changed["inside"][lat < -30].max() > 0.3      # ... and a ceiling below it
    assert changed["outside"][lat < 0].max() < 1e-9      # the outside is the other way round
    assert changed["outside"][lat > 30].max() > 0.3
    with pytest.raises(ValueError):
        imaging.limit_overhang_slope(T, phis, radius, True, reach)   # the old boolean argument


def test_split_moves_the_relief_across_the_equator():
    b = SPLIT_BLEND_DEG
    lat = np.array([-50.0, -b, -b / 2, 0.0, b / 2, b, 50.0])
    share = imaging.relief_share("split", np.radians(lat))
    assert np.allclose(share[[0, 1]], 1.0) and np.allclose(share[[5, 6]], 0.0)   # outside below, inside above
    assert share[3] == pytest.approx(0.5) and (np.diff(share) <= 0).all()
    assert np.allclose(share, [split_share(math.radians(x)) for x in lat])
    assert not imaging.relief_share("inside", np.radians(lat)).any()
    assert imaging.relief_share("outside", np.radians(lat)).all()


def test_outside_first_shares_a_worst_case_picture_between_the_two_faces():
    """Random black-and-white noise is the worst photo there is.  All of it on the outside
    hangs in mid-air in the half printed first; shared 'outside first', neither face is
    closer to flat than the design allows, and the wall is still as thick as the picture
    asks everywhere except right at the equator, where both faces are at their limit."""
    rng = np.random.default_rng(1)
    p = Params()
    t_frame, t_min, t_max, R = p.t_frame, p.t_min, p.t_max, p.diameter / 2
    r_out, r_in = outside_first_radii(p)
    assert r_out == pytest.approx(R - t_max + t_min) and r_in == pytest.approx(R - 2 * t_max + t_min)
    phis = np.radians(np.linspace(-60, 60, 337))
    lat = np.degrees(phis)
    T = np.full((len(phis), 60), t_frame)
    band = np.abs(lat) < 50
    T[band] = rng.choice([t_min, t_max], size=(int(band.sum()), 60))
    extra = T - t_frame
    ro, ri = R - t_max + t_frame, R - t_max                  # the plain shell, relief outside
    assert _steepest_of(ro + extra, ri + 0 * extra, phis) < 0            # tips below their roots
    out, inside = imaging.outside_first(extra, phis, r_out, r_in)
    assert _steepest_of(ro + out, ri - inside, phis) >= EDGE_RAMP_MIN_DEG - 0.3
    # the outside - the face you see - is held to the angle of the standard shoulder
    assert OUTSIDE_RAMP_MIN_DEG == 30.0 > EDGE_RAMP_MIN_DEG
    assert _steepest_face(ro + out, phis, 1.0) >= OUTSIDE_RAMP_MIN_DEG - 0.3
    built = t_frame + out + inside
    assert built.min() >= t_min - 1e-9 and built.max() <= t_max + 1e-9
    wrong = np.abs(built - T).max(axis=1) > 1e-9
    assert lat[wrong].max() < 20.0                           # exact wherever the inside is free
    assert lat[wrong].min() > -24.0                          # ... or the outside is
    assert not out[[0, -1]].any() and not inside[[0, -1]].any()     # the rows that join the shell
    assert not inside[lat < -31].any()                       # the half printed last: all outside
    assert np.allclose(out[lat < -31], extra[lat < -31])


def test_outside_first_leaves_everything_it_can_on_the_outside():
    p = Params()
    radii = outside_first_radii(p)
    phis = np.radians(np.linspace(-60, 60, 481))
    lat = np.degrees(phis)[:, None]
    # a smooth picture (fading out towards the plain rows): nothing goes inside
    fade = np.clip((52.0 - np.abs(lat)) / 12.0, 0.0, 1.0)
    smooth = 0.5 * np.sin(np.linspace(0, 9, len(phis)))[:, None] * fade * np.ones((1, 3))
    out, inside = imaging.outside_first(smooth, phis, *radii)
    assert np.allclose(out, smooth) and not inside.any()

    def edge(latitude: float, thick_above_on_printer: bool) -> np.ndarray:
        e = np.where((lat < latitude) == thick_above_on_printer, p.t_max, p.t_min) - p.t_frame
        e[np.abs(lat[:, 0]) > 55] = 0.0
        return e

    # an edge where the wall gets THINNER going up on the printer prints as it is, anywhere
    for where in (-35.0, 0.0, 35.0):
        e = edge(where, False)
        out, inside = imaging.outside_first(e, phis, *radii)
        keep = np.abs(lat[:, 0]) < 50                        # (the made-up edges at +-55 aside)
        assert np.allclose(out[keep], e[keep]) and not inside[keep].any(), where
    # one where it thickens going up, in the half printed last: the outside faces up, so again
    e = edge(-35.0, True)
    out, inside = imaging.outside_first(e, phis, *radii)
    assert np.allclose(out[np.abs(lat[:, 0]) < 50], e[np.abs(lat[:, 0]) < 50])
    assert not inside[np.abs(lat[:, 0]) < 50].any()
    # the same edge in the half printed first: the outside can only ramp, centred on the edge,
    # and the inside makes up the difference exactly - half as a dip before it, half as a
    # bump after it
    e = edge(35.0, True)
    out, inside = imaging.outside_first(e, phis, *radii)
    near = np.abs(lat[:, 0] - 35.0) < 15
    assert np.allclose((out + inside)[near], e[near])
    step = p.t_max - p.t_min
    assert 0.4 * step < inside[near].max() < 0.55 * step and 0.4 * step < -inside[near].min() < 0.55 * step
    assert abs(inside[near].sum()) < 0.02 * np.abs(inside[near]).sum()
    ramp = np.abs(out[near] - e[near]).max(axis=1) > 1e-9
    arc = np.radians(np.ptp(lat[near][ramp])) * (p.diameter / 2)
    assert 3.0 < arc < 5.5                                   # 2 mm of wall gained over about 4 mm
    # at the equator the inside is limited as well: the edge is spread over half a millimetre
    e = edge(0.0, True)
    out, inside = imaging.outside_first(e, phis, *radii)
    off = np.abs(out + inside - e).max(axis=1)
    assert 0.1 < off.max() <= step / 2 + 1e-9             # the ramp is centred on the edge
    assert np.radians(np.ptp(lat[off > 1e-9])) * (p.diameter / 2) < 0.8


def _thinnest_across(outer: np.ndarray, inside: np.ndarray, phis: np.ndarray, lon_step: float,
                     radius: float) -> float:
    """Smallest distance in space between the two faces of a wall given the way
    imaging.keep_true_thickness takes it: the outside as a height above the plain inside
    sphere of this radius, the inside as a depth below it (rows = latitudes, columns
    lon_step apart)."""
    from scipy.spatial import cKDTree

    lam = np.arange(outer.shape[1]) * lon_step

    def points(height: np.ndarray) -> np.ndarray:
        rho = radius + height
        cp, sp = np.cos(phis)[:, None], np.sin(phis)[:, None]
        return np.stack([rho * cp * np.cos(lam), rho * cp * np.sin(lam), rho * sp], axis=-1).reshape(-1, 3)

    return float(cKDTree(points(outer)).query(points(-inside))[0].min())


def test_sharing_an_edge_never_leaves_the_wall_too_thin_across():
    """Where the inside makes room for an outside that is still on its ramp, both faces move
    outwards together.  Beside a column that has no such edge - or along a steep ramp near
    the equator - the wall would then be a thin slanted web, far thinner across than along
    the radius.  The inside gives way there, and only there."""
    p = Params()
    radii = outside_first_radii(p)
    wall, least = p.t_frame, least_true_wall(p)
    assert least == pytest.approx(TRUE_WALL_SHARE * p.t_min) and 0.7 < least < p.t_min
    n_lon = 240
    lon_step = 2 * math.pi / 1005                    # the column spacing of the standard model
    phis = np.radians(np.linspace(-60, 60, 961))     # 0.125 deg: about 0.08 mm a row
    lat = np.degrees(phis)[:, None]
    bright = p.t_min - p.t_frame

    def picture(edge_at: float) -> np.ndarray:
        """Bright everywhere, with a dark block above (on the printer) an edge at this
        latitude in half of the columns."""
        e = np.full((len(phis), n_lon), bright)
        e[(lat[:, 0] < edge_at) & (lat[:, 0] > edge_at - 12.0), :n_lon // 2] = p.t_max - p.t_frame
        e[np.abs(lat[:, 0]) > 52] = 0.0
        return e

    for edge_at in (42.0, 12.0):                     # a gentle ramp, and a steep one near the equator
        e = picture(edge_at)
        out, raw = imaging.outside_first(e, phis, *radii)
        assert raw.min() < -0.5                      # a deep dip before the edge ...
        thin = _thinnest_across(wall + out, raw, phis, lon_step, radii[1])
        assert thin < 0.6 * least, (edge_at, thin)   # ... and a web far thinner than any wall should be
        out2, kept = imaging.outside_first(e, phis, *radii, wall=wall, lon_step=lon_step, least=least)
        assert np.array_equal(out2, out)             # the outside - the picture - is not touched
        assert _thinnest_across(wall + out, kept, phis, lon_step, radii[1]) >= least - 0.02, edge_at
        assert (kept >= raw - 1e-12).all()           # dips are filled, nothing is dug out
        built, asked = wall + out + kept, wall + e
        assert built.min() >= p.t_min - 1e-9         # the wall only ever gains from this
        if edge_at > 30:                             # (near the equator the inside is slope-limited
            assert (built >= asked - 1e-9).all()     #  too, and eases an extreme edge either way)
        changed = np.abs(kept - raw) > 1e-9
        assert 0 < changed.mean() < 0.02             # a sliver of the surface
        # only rows near the dip, and (for the gentle ramp) only columns beside the plain half
        rows = np.nonzero(changed.any(axis=1))[0]
        assert lat[rows, 0].min() > edge_at - 1.0 and lat[rows, 0].max() < edge_at + 8.0
        if edge_at > 30:        # (a wall displaced along so gentle a ramp is thick enough as it is)
            cols = np.nonzero(changed.any(axis=0))[0]
            reach = least / (math.cos(math.radians(edge_at + 8)) * radii[1] * lon_step) + 2
            near_seam = np.minimum(np.abs(cols - n_lon // 2 + 0.5), np.minimum(cols + 0.5, n_lon - cols - 0.5))
            assert near_seam.max() <= reach, (near_seam.max(), reach)
            # in the middle of the edge the wall is still exactly what the picture asks
            mid = slice(n_lon // 8, 3 * n_lon // 8)
            assert np.allclose(built[:, mid][np.abs(lat[:, 0] - edge_at) < 8],
                               asked[:, mid][np.abs(lat[:, 0] - edge_at) < 8])

    # nothing to give way to: the very same array comes back
    plain = np.zeros((len(phis), 12))
    assert imaging.keep_true_thickness(wall + plain, plain, phis, lon_step, radii[1], least) is plain
    bump = plain.copy()
    bump[400:420] = 0.5                               # the inside standing IN is never the problem
    assert imaging.keep_true_thickness(wall + plain, bump, phis, lon_step, radii[1], least) is bump


@pytest.mark.parametrize("side", SIDES)
def test_no_part_of_the_finished_wall_is_too_thin_across(harsh, side):
    """The whole model, with the hardest photo there is: measured in space, the two faces
    of the wall are nowhere closer than nine tenths of the thinnest wall - and a horizontal
    cut, which is what the printer makes, is a ring that survives losing 0.3 mm all round
    (so it is everywhere wide enough for two lines of a 0.4 mm nozzle)."""
    from scipy.spatial import cKDTree
    m3 = pytest.importorskip("manifold3d")

    p = with_photos(Params(relief_side=side, smoothing=0.0), harsh)
    for slot in p.images:
        slot.auto_levels = False
    lay = derive(p)
    photos, _ = imaging.photos_for(p)
    body = geometry.build_body(p, lay, photos, pitch=0.3, dense=True)
    outer, inner = body.vertices[:body.n_outer], body.vertices[body.n_outer:]
    radius = np.linalg.norm(inner, axis=1)
    in_band = np.abs(np.arcsin(np.clip(inner[:, 2] / radius, -1, 1))) < lay.band_half
    nearest = cKDTree(outer).query(inner[in_band])[0]
    assert nearest.min() >= least_true_wall(p) - 0.03, nearest.min()

    solid = m3.Manifold(m3.Mesh64(vert_properties=np.ascontiguousarray(body.vertices),
                                  tri_verts=np.ascontiguousarray(body.faces, dtype=np.uint64)))
    assert solid.status() == m3.Error.NoError
    half = lay.R * math.sin(lay.band_half)
    for z in np.linspace(-half + 1.0, half - 1.0, 23):
        eroded = solid.slice(float(z)).offset(-0.3, m3.JoinType.Round, 2.0, 16)
        assert any(len(part.to_polygons()) >= 2 for part in eroded.decompose()), (side, z)


def test_relief_on_the_inside_looks_soft_and_the_preview_says_so(harsh):
    """The lit preview is coloured by the thickness the wall LOOKS, which blurs whatever
    relief is on the inside of the wall - the way the first printed ornament looked."""
    def sharpness(side: str) -> tuple[float, float]:
        p = with_photos(Params(relief_side=side, smoothing=0.0), harsh)
        lay = derive(p)
        photos, _ = imaging.photos_for(p)
        body = geometry.build_body(p, lay, photos, pitch=0.3, dense=True)
        n = body.n_lon
        assert body.lit is not None and body.lit.shape == body.thickness.shape
        real = body.thickness[1:1 + body.outer_rows * n].reshape(body.outer_rows, n)
        seen = body.lit[1:1 + body.outer_rows * n].reshape(body.outer_rows, n)
        outer = body.vertices[1:1 + body.outer_rows * n].reshape(body.outer_rows, n, 3)[:, 0]
        lat = np.degrees(np.arcsin(outer[:, 2] / np.linalg.norm(outer, axis=1)))
        upper, lower = (lat > 12) & (lat < 48), (lat < -12) & (lat > -48)

        def kept(rows: np.ndarray) -> float:               # 1 = every step as tall as in the wall
            return float(np.abs(np.diff(seen[rows], axis=0)).mean()
                         / np.abs(np.diff(real[rows], axis=0)).mean())

        assert np.isnan(body.lit).sum() == np.isnan(body.thickness).sum()
        return kept(upper), kept(lower)

    up, low = sharpness("outside")
    assert up > 0.9 and low > 0.99                           # sharp all over
    up, low = sharpness("split")
    assert up < 0.5 and low > 0.99                           # the half that came out blurred
    up, low = sharpness("inside")
    assert up < 0.5 and low < 0.5


def test_softening_of_inside_relief():
    tau = render.ATTENUATION_MM
    flat = np.zeros((40, 30))
    assert np.allclose(render.soften_inside(flat, 0.25, np.full(40, 0.25)), 0.0)
    # a 1.5 mm dip (a bright dot): about half of its extra light is left at its centre
    dip = np.zeros((200, 60))
    dip[97:103] = -1.4
    soft = render.soften_inside(dip, 0.25, np.full(200, 0.25))
    gain, full = np.exp(-soft[100, 30] / tau) - 1.0, np.exp(1.4 / tau) - 1.0
    assert 0.4 < gain / full < 0.7
    assert np.abs(soft[:40]).max() < 1e-6                   # far away nothing changes
    # light is spread out, not lost
    assert np.exp(-soft / tau).sum() == pytest.approx(np.exp(-dip / tau).sum(), rel=1e-6)
    # a wide area is unchanged in its middle
    wide = np.zeros((200, 60))
    wide[60:140] = 1.0
    assert render.soften_inside(wide, 0.25, np.full(200, 0.25))[100, 30] == pytest.approx(1.0, abs=1e-3)
    # columns that are closer together (towards the top of the ball) blur over more of them
    spot = np.zeros((60, 200))
    spot[:, 99:101] = -1.0
    close = render.soften_inside(spot, 0.25, np.full(60, 0.15))
    apart = render.soften_inside(spot, 0.25, np.full(60, 0.30))
    assert close[30, 99] > apart[30, 99] + 0.05             # shallower = more blurred


def test_checklist_says_which_side_looks_sharp(gradient):
    def relief_checks(**kwargs):
        p = with_photos(Params(**kwargs), gradient)
        return checks.analyze(p, derive(p), 30.0)

    assert any(c.level == checks.OK and c.title.startswith("Sharp picture") for c in relief_checks())
    soft = [c for c in relief_checks(relief_side="split") if "soft" in c.title]
    assert len(soft) == 1 and soft[0].level == checks.WARN and "Outside" in soft[0].detail
    soft = [c for c in relief_checks(relief_side="inside") if "soft" in c.title]
    assert len(soft) == 1 and soft[0].level == checks.WARN and "eased" in soft[0].detail
    titles = [c.title for c in relief_checks(relief_side="inside", safe_edges=False)]
    assert "The pictures will look soft" in titles and "Details may print in mid-air" in titles
    titles = [c.title for c in relief_checks(safe_edges=False)]
    assert "Details may print in mid-air" in titles and not any(t.startswith("Sharp") for t in titles)


def test_files_with_an_old_default_relief_move_to_outside():
    """"inside" was the only choice before 1.1 and "split" the default of 1.1 and 1.2.  Both
    turned out blurred where the relief is inside the wall, so files that carry either as a
    default open with the relief outside; a deliberate later choice is kept."""
    old_default = {"params": {"relief_side": "inside", "diameter": 90}}
    assert params_from_file(old_default).relief_side == "outside"
    assert params_from_file({"version": "1.0.0", **old_default}).relief_side == "outside"
    assert params_from_file(old_default).diameter == 90
    assert params_from_file({"version": "1.1.0", **old_default}).relief_side == "inside"
    assert params_from_file({"version": "1.3.0", **old_default}).relief_side == "inside"
    for version in ("1.1.0", "1.2.0", "1.2.9"):
        assert params_from_file({"version": version, "params": {"relief_side": "split"}}).relief_side == "outside"
    assert params_from_file({"params": {"relief_side": "split"}}).relief_side == "outside"
    assert params_from_file({"version": "1.3.0", "params": {"relief_side": "split"}}).relief_side == "split"
    assert params_from_file({"params": {"relief_side": "outside"}}).relief_side == "outside"
    assert Params().relief_side == "outside"
    assert list(RELIEF_SIDES)[0] == "outside"            # and it is the first choice offered


def test_files_with_the_old_default_gap_move_to_the_new_one():
    assert Params().gap_deg == 12.0
    assert params_from_file({"params": {"gap_deg": 6.0}}).gap_deg == 12.0            # first version
    assert params_from_file({"version": "1.1.0", "params": {"gap_deg": 6.0}}).gap_deg == 12.0
    assert params_from_file({"version": "1.2.0", "params": {"gap_deg": 6.0}}).gap_deg == 6.0
    # a gap somebody chose is kept, whatever the version
    assert params_from_file({"version": "1.0.0", "params": {"gap_deg": 8.0}}).gap_deg == 8.0
    assert params_from_file({"params": {"gap_deg": 0.0}}).gap_deg == 0.0


# ---------------------------------------------------------------------------
# seam line
# ---------------------------------------------------------------------------
def _azimuth_from_back(vertices: np.ndarray) -> np.ndarray:
    return np.abs(np.degrees(np.arctan2(vertices[:, 0], vertices[:, 1])))


@pytest.mark.parametrize("side", SIDES)
def test_seam_line_runs_down_the_back_divider_on_both_faces(gradient, side):
    p = with_photos(Params(relief_side=side), gradient)
    build = geometry.build_ornament(p, pitch=PITCH)
    lay, v, f = build.layout, build.vertices, build.faces
    seam = build.seam_faces
    assert seam is not None and seam.dtype == bool and len(seam) == len(f) and 0 < seam.sum() < len(f) / 20
    half = geometry.seam_line_half_deg(lay, PITCH)
    marked = v[np.unique(f[seam])]
    assert _azimuth_from_back(marked).max() <= half + 1e-9
    # the line is well inside the divider (plain frame), with the slicer's reach to spare
    assert half < 6.0 and geometry.seam_fits_divider(lay, PITCH)

    _assert_seam_line_on_both_faces(build, p)


def _assert_seam_line_on_both_faces(build, p: Params, step: float = 0.12) -> None:
    """At every printing height through the pictures there is marked surface on
    the outside AND on the inside of the wall, so both loops of a layer find the line."""
    lay = build.layout
    tri = build.vertices[build.faces[build.seam_faces]]
    z_lo, z_hi = tri[:, :, 2].min(axis=1), tri[:, :, 2].max(axis=1)
    radius = np.hypot(tri[:, :, 0], tri[:, :, 1]).mean(axis=1)
    reach = lay.R * math.sin(lay.band_half)
    for z in np.arange(lay.z_top - reach, lay.z_top + reach, step):
        r = radius[(z_lo <= z) & (z <= z_hi)]
        # the two faces are at least the thinnest wall apart
        assert len(r) >= 2 and r.max() - r.min() > 0.9 * p.t_min, z


def test_seam_line_is_only_marked_on_the_print_orientation(gradient):
    p = with_photos(Params(), gradient)
    assert geometry.build_ornament(p, pitch=PITCH, orientation="hanging").seam_faces is None
    # the same triangles whichever way they are found: straight from the helper ...
    build = geometry.build_ornament(p, pitch=PITCH)
    again = geometry.seam_line_faces(build.vertices, build.faces,
                                     geometry.seam_line_half_deg(build.layout, PITCH))
    assert (again == build.seam_faces).all()
    # ... and nothing at the front or the sides is marked
    centroid = build.vertices[build.faces[build.seam_faces]].mean(axis=1)
    assert (centroid[:, 1] > 0).all()


def test_seam_line_narrows_with_the_gap_but_keeps_a_mesh_column():
    pitch = 0.25
    column = 360.0 / geometry.longitude_count(40.0, pitch)
    wide = derive(Params(gap_deg=12.0))
    assert geometry.seam_line_half_deg(wide, pitch) == pytest.approx(geometry.SEAM_LINE_DEG)
    assert geometry.seam_fits_divider(wide, pitch)
    narrow = derive(Params(gap_deg=3.0))
    half = geometry.seam_line_half_deg(narrow, pitch)
    assert 1.05 * column < half < geometry.SEAM_LINE_DEG and geometry.seam_fits_divider(narrow, pitch)
    for gap in (2.0, 0.0):
        tight = derive(Params(gap_deg=gap))
        assert geometry.seam_line_half_deg(tight, pitch) == pytest.approx(1.05 * column)
        assert not geometry.seam_fits_divider(tight, pitch)
    # the narrowest gap that works is what the checklist asks for
    need = geometry.seam_min_gap_deg(wide, pitch)
    assert 2.0 < need < 3.0
    assert geometry.seam_fits_divider(derive(Params(gap_deg=math.ceil(need * 2) / 2)), pitch)
    # one whole column of triangles is always inside the narrowest line, wherever the mesh
    # columns fall (the back is on a column for some column counts, between two for others)
    counts = set()
    for mesh_pitch in (0.60, 0.61, 0.62, 0.63):
        p = Params(gap_deg=0.0, hanger="none")
        build = geometry.build_ornament(p, pitch=mesh_pitch)
        counts.add(geometry.longitude_count(build.layout.R, mesh_pitch) % 4)
        _assert_seam_line_on_both_faces(build, p, step=0.5)
    assert counts == {0, 2}


def test_checklist_says_where_the_seam_goes():
    def seam_row(p: Params) -> checks.Check:
        rows = [c for c in checks.analyze(p, derive(p)) if c.title.startswith("Seam")]
        assert len(rows) == 1
        return rows[0]

    assert seam_row(Params()).level == checks.OK and seam_row(Params()).title == "Seam hidden"
    assert seam_row(Params(gap_deg=3.0)).level == checks.OK
    tight = seam_row(Params(gap_deg=1.0))
    assert tight.level == checks.WARN and tight.title == "Seam touches the pictures"
    assert "3 deg or more" in tight.detail
    none = seam_row(Params(gap_deg=0.0))
    assert none.level == checks.WARN and none.title == "Seam between two pictures"


def test_3mf_carries_the_seam_line_and_stl_does_not(tmp_path):
    import zipfile

    build = geometry.build_ornament(Params(), pitch=1.2)
    painted = str(tmp_path / "painted.3mf")
    export.write_mesh(painted, build.vertices, build.faces, build.seam_faces)
    with zipfile.ZipFile(painted) as zf:
        triangles = [ln for ln in zf.read("3D/3dmodel.model").decode().splitlines()
                     if ln.startswith("<triangle ")]
    assert len(triangles) == len(build.faces)
    marked = np.array(['paint_seam="4"' in ln for ln in triangles])
    assert (marked == build.seam_faces).all() and marked.any()
    assert all(ln.endswith("/>") for ln in triangles)

    plain = str(tmp_path / "plain.3mf")
    export.write_mesh(plain, build.vertices, build.faces)
    with zipfile.ZipFile(plain) as zf:
        assert "paint_seam" not in zf.read("3D/3dmodel.model").decode()
    stl = str(tmp_path / "plain.stl")
    export.write_mesh(stl, build.vertices, build.faces, build.seam_faces)       # ignored, harmless
    assert os.path.getsize(stl) == 84 + 50 * len(build.faces)


def test_zoom_out_leaves_a_frame_border(gradient):
    p = with_photos(Params(n_images=2, gap_deg=0.0), gradient)
    for slot in p.images:
        slot.zoom = 0.5
    _lay, _lam, _phis, T = _thickness_grid(p)
    assert np.isclose(T, p.t_frame).mean() > 0.3


def test_missing_photo_is_reported_not_fatal(tmp_path):
    p = Params(n_images=2)
    p.images[0].path = str(tmp_path / "does_not_exist.jpg")
    build = geometry.build_ornament(p, pitch=PITCH)
    assert len(build.problems) == 2
    assert checks.verify_mesh(build.vertices, build.faces)["watertight"]


# ---------------------------------------------------------------------------
# mesh validity over the whole parameter space
# ---------------------------------------------------------------------------
MATRIX = list(itertools.product([1, 2, 3, 4], SIDES, ["two", "one", "none"]))


@pytest.mark.parametrize("n,side,hanger", MATRIX)
def test_mesh_is_a_valid_solid(gradient, n, side, hanger):
    p = with_photos(Params(n_images=n, relief_side=side, hanger=hanger), gradient)
    build = geometry.build_ornament(p, pitch=PITCH)
    report = checks.verify_mesh(build.vertices, build.faces)
    assert report["watertight"], report
    assert report["finite"] and report["degenerate_faces"] == 0, report
    assert report["volume_mm3"] > 5000
    lay = build.layout
    lo, hi = build.vertices.min(axis=0), build.vertices.max(axis=0)
    assert abs(lo[2]) < 1e-6, "the cap must sit exactly on the bed"
    assert abs(hi[2] - lay.height) < 1e-6
    assert abs(build.stats["size_mm"][2] - lay.height) < 1e-6


@pytest.mark.parametrize("overrides", [
    dict(diameter=50, cap_diameter=18, hole_diameter=8),
    dict(diameter=150, cap_diameter=45, hole_diameter=30.5, light_preset="c9"),
    dict(diameter=60, hole_diameter=6, cap_diameter=9, cap_height=6),      # plain tube cap
    dict(flare_angle=25), dict(flare_angle=50, cap_diameter=40),
    dict(t_min=0.4, t_max=4.0, t_frame=0.8), dict(t_frame=4.0),
    dict(band_height_deg=40, gap_deg=40), dict(band_height_deg=120, gap_deg=0),
    dict(n_images=1, gap_deg=0), dict(cap_wall=4.0, cap_height=20),
    dict(hanger_hole=6.0, cap_diameter=12, hole_diameter=6),
    dict(relief_side="outside", t_frame=0.8, t_max=4.0),
    dict(relief_side="inside", t_frame=4.0, t_min=0.4, band_height_deg=120),
    dict(relief_side="inside", diameter=50, cap_diameter=18, hole_diameter=8),
])
def test_extreme_settings_still_build(gradient, overrides):
    p = with_photos(Params(**overrides), gradient)
    lo, hi = cap_diameter_range(p)
    p.cap_diameter = min(hi, max(lo, p.cap_diameter))
    build = geometry.build_ornament(p, pitch=PITCH)
    report = checks.verify_mesh(build.vertices, build.faces)
    assert report["watertight"] and report["finite"] and report["volume_mm3"] > 0, report


def test_preview_volume_matches_the_real_mesh(gradient):
    p = with_photos(Params(), gradient)
    preview = geometry.build_preview(p)
    build = geometry.build_ornament(p, pitch=0.3)
    assert abs(preview.volume_cm3 - build.stats["volume_cm3"]) / build.stats["volume_cm3"] < 0.01
    assert len(preview.body.thickness) == len(preview.body.vertices)
    normals = geometry.body_normals(preview.body)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
    # outer normals point away from the centre on the sphere part
    outer = preview.body.vertices[1:2000]
    assert (np.einsum("ij,ij->i", outer, normals[1:2000]) > 0).all()


# ---------------------------------------------------------------------------
# printability of the shape itself (no picture, so the relief does not interfere)
# ---------------------------------------------------------------------------
def _down_faces(v: np.ndarray, faces: np.ndarray):
    """Downward-facing triangles of a mesh in print orientation:
    (angle from horizontal in degrees, centroid z, area)."""
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    keep = area > 1e-9
    n, area = n[keep] / (2 * area[keep, None]), area[keep]
    z = ((a + b + c) / 3)[keep, 2]
    down = n[:, 2] < -1e-6
    angle = np.degrees(np.arccos(np.clip(-n[down, 2], -1, 1)))     # 0 = flat ceiling
    return angle, z[down], area[down]


def _overhangs(p: Params):
    """The same for the plain shell (no picture, no hanger loops)."""
    lay = derive(p)
    body = geometry.build_body(p, lay, None, pitch=0.8)
    return (lay,) + _down_faces(geometry.to_print_orientation(body.vertices, lay), body.faces)


@pytest.fixture()
def harsh(tmp_path) -> str:
    """The worst photo for a printer: big blocks of pure black and pure white,
    so every edge is as tall and as sudden as an edge can be."""
    rng = np.random.default_rng(5)
    blocks = rng.choice([0, 255], size=(40, 40))
    return make_image(tmp_path, "harsh.png", np.kron(blocks, np.ones((12, 12))))


@pytest.mark.parametrize("side", SIDES)
def test_nothing_starts_in_mid_air_or_overhangs_too_far(harsh, side):
    """The finished model, as it stands on the printer: no point of it is lower than
    everything it is joined to (what a slicer calls a floating region), and no
    downward face is closer to flat than the shell's own shallowest surface or
    the limit for the picture relief - whichever is lower."""
    p = with_photos(Params(relief_side=side, smoothing=0.0), harsh)
    for slot in p.images:
        slot.auto_levels = False
    build = geometry.build_ornament(p, pitch=0.3)
    lay = build.layout
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    angle, z, _area = _down_faces(build.vertices, build.faces)
    bridge = (np.abs(z - (lay.z_top + lay.z_f)) < 1e-6) & (angle < 1.0)
    rest = ~((z < 1e-6) | bridge)
    assert angle[rest].min() >= min(EDGE_RAMP_MIN_DEG, p.flare_angle, CLOSURE_ANGLE_DEG) - 0.5
    if side == "split":
        # The relief only sits on faces that point up, so the one place it can overhang at
        # all is next to the equator, where the wall is upright (as on any standing lithophane).
        steep = rest & (angle < min(p.flare_angle, CLOSURE_ANGLE_DEG) - 0.5)
        assert np.abs(z[steep] - lay.z_top).max() < lay.R * math.sin(math.radians(22.0))


@pytest.mark.parametrize("side", ["inside", "outside"])
def test_without_print_safe_edges_one_sided_relief_hangs(harsh, side):
    """What the slicer warned about: with the relief on one side only and the edges
    left as they are, dark details hang from the leaning wall.  The check finds them
    (the app warns on export).  Print-safe edges put it right."""
    p = with_photos(Params(relief_side=side, smoothing=0.0, safe_edges=False), harsh)
    build = geometry.build_ornament(p, pitch=0.3)
    tips = checks.hanging_tips(build.vertices, build.faces)
    assert len(tips) > 20
    # ... all of them in the half where that side of the wall faces down
    middle = build.layout.z_top            # printer height of the equator
    assert (tips[:, 2] > middle).all() if side == "inside" else (tips[:, 2] < middle).all()
    assert any(c.level == checks.WARN and "mid-air" in c.title for c in checks.analyze(p, build.layout))
    p.safe_edges = True
    build = geometry.build_ornament(p, pitch=0.3)
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    assert not any("mid-air" in c.title for c in checks.analyze(p, build.layout))
    p.safe_edges, p.relief_side = False, "split"        # on both sides nothing can hang even then
    build = geometry.build_ornament(p, pitch=0.3)
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    assert not any("mid-air" in c.title for c in checks.analyze(p, build.layout))


def test_hanging_tip_check_tells_a_tip_from_a_dimple():
    m3 = pytest.importorskip("manifold3d")
    slab = m3.Manifold.cube((10.0, 10.0, 2.0))                       # lies on the bed
    spike = m3.Manifold.sphere(1.0, 4).translate((20.0, 5.0, 6.0))    # an octahedron in mid-air
    dimple = m3.Manifold.sphere(1.0, 4).translate((5.0, 5.0, 2.0))    # ... and one cut out of the top

    def arrays(solid):
        mesh = solid.to_mesh()
        return (np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
                np.asarray(mesh.tri_verts, dtype=np.int64))

    assert len(checks.hanging_tips(*arrays(slab))) == 0
    assert len(checks.hanging_tips(*arrays(slab - dimple))) == 0     # lowest point of a hollow: harmless
    tips = checks.hanging_tips(*arrays(slab + spike))
    assert len(tips) == 1 and np.allclose(tips[0], (20.0, 5.0, 5.0))


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("flare", [25.0, 30.0, 40.0])
def test_no_overhang_is_shallower_than_designed(side, flare):
    p = Params(relief_side=side, flare_angle=flare)
    lay, angle, z, area = _overhangs(p)
    on_bed = z < 1e-6
    ceiling_z = lay.z_top + lay.z_f                 # the bridged ceiling at the closed end
    bridge = (np.abs(z - ceiling_z) < 1e-6) & (angle < 1.0)
    rest = ~(on_bed | bridge)
    limit = min(flare, CLOSURE_ANGLE_DEG)
    assert angle[rest].min() >= limit - 0.6, angle[rest].min()
    # the only flat ceiling is the small bridge, and it is the size the layout says
    assert math.isclose(area[bridge].sum(), math.pi * lay.r_f ** 2, rel_tol=0.02)
    assert 2 * lay.r_f < checks.MAX_CLEAN_BRIDGE_MM


@pytest.mark.parametrize("side", ["inside", "outside"])
def test_frame_ramps_instead_of_stepping_over_the_picture_edge(tmp_path, side):
    """A white photo next to a thick frame (or a black one next to a thin frame)
    is the worst case for the ledge that would print over thin air."""
    shade = 255 if side == "inside" else 0
    photo = make_image(tmp_path, "flat.png", np.full((200, 200), shade))
    p = Params(relief_side=side, smoothing=0.0, t_frame=2.2 if side == "inside" else 1.2)
    for slot in p.images:
        slot.path, slot.auto_levels = photo, False
    lay = derive(p)
    photos, _ = imaging.photos_for(p)
    pitch = 0.5
    n_lon = geometry.longitude_count(lay.R, pitch)
    lam = np.arange(n_lon) * (2 * math.pi / n_lon)
    inside = side == "inside"
    phis, band = geometry._relief_rows(lay, pitch, -lay.phi_c)
    T = np.full((len(phis), n_lon), p.t_frame)
    T[band] = imaging.thickness_band(p, lay, lam, phis[band], photos, pitch)
    step_before = np.abs(T[band.start] - p.t_frame).max()
    assert step_before > 0.9                      # there really is a big step to deal with
    geometry._ramp_frame_edge(T, phis, band, lay, p, inside, pitch)
    slope = geometry.edge_ramp_slope(lay)
    arc = np.diff(phis) * lay.R                    # distance between neighbouring rows
    if inside:      # towards the closed end (lower rows) the wall may only thicken gently
        rows = range(1, band.start)
        rise = [(T[j] - T[j + 1]).max() / arc[j] for j in rows]
    else:           # towards the cap (higher rows) the wall may only thin gently
        rows = range(band.stop - 1, len(phis) - 2)
        rise = [(T[j] - T[j + 1]).max() / arc[j] for j in rows]
    assert max(rise) <= slope + 1e-6, (max(rise), slope)
    assert np.allclose(T[0 if inside else -1], p.t_frame)     # the joining row is untouched
    # and the picture itself was not altered
    assert np.allclose(T[band], imaging.thickness_band(p, lay, lam, phis[band], photos, pitch))


@pytest.mark.parametrize("side", SIDES)
def test_wall_is_exactly_as_thick_as_the_picture_asks(gradient, side):
    """Measure the real wall of the mesh along rays from the centre: whichever
    surface carries the relief, outside radius minus inside radius is the
    thickness the picture asked for (this is what the lit preview is coloured by)."""
    p = with_photos(Params(relief_side=side, smoothing=0.0), gradient)
    lay = derive(p)
    photos, _ = imaging.photos_for(p)
    body = geometry.build_body(p, lay, photos, pitch=0.5, dense=True)
    n = body.n_lon
    outer = body.vertices[1:1 + body.outer_rows * n].reshape(body.outer_rows, n, 3)
    known = body.thickness[1:1 + body.outer_rows * n].reshape(body.outer_rows, n)
    inner = body.vertices[body.n_outer + 1:body.n_outer + 1 + body.inner_rows * n].reshape(
        body.inner_rows, n, 3)
    r_out, r_in = np.linalg.norm(outer, axis=2), np.linalg.norm(inner, axis=2)
    lat_out = np.arcsin(outer[:, 0, 2] / r_out[:, 0])
    lat_in = np.arcsin(inner[:, 0, 2] / r_in[:, 0])
    checked = 0
    for j, lat in enumerate(lat_in):
        if abs(lat) > lay.band_half:
            continue                                # frame, cone and shoulder rows
        k = int(np.argmin(np.abs(lat_out - lat)))   # the outside row at the same latitude
        assert abs(lat_out[k] - lat) < 1e-9
        assert np.allclose(r_out[k] - r_in[j], known[k], atol=1e-9)
        checked += 1
    assert checked > 20
    band = np.abs(lat_out) < lay.band_half
    assert known[band].min() >= p.t_min - 1e-6 and known[band].max() <= max(p.t_max, p.t_frame) + 1e-6
    assert known[band].max() - known[band].min() > 0.7 * (p.t_max - p.t_min)
    # each surface is a plain sphere wherever the other one carries all of the relief
    if side != "outside":
        above = lat_out > math.radians(SPLIT_BLEND_DEG) + 1e-9 if side == "split" else band
        assert np.allclose(r_out[above & (lat_out < lay.phi_t - 1e-9)], lay.ro_frame)
    if side != "inside":
        below = (lat_in < -math.radians(SPLIT_BLEND_DEG) - 1e-9) if side == "split" else \
            (np.abs(lat_in) < lay.band_half)
        assert np.allclose(r_in[below & (lat_in > -lay.phi_c + 1e-9)], lay.ri_frame)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def test_stl_and_3mf_round_trip(gradient, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    p = with_photos(Params(), gradient)
    build = geometry.build_ornament(p, pitch=PITCH)
    for name in ("out.stl", "out.3mf"):
        path = str(tmp_path / name)
        export.write_mesh(path, build.vertices, build.faces)
        loaded = trimesh.load(path, force="mesh")
        assert len(loaded.faces) == len(build.faces)
        assert loaded.is_watertight
        assert abs(loaded.volume - build.stats["volume_cm3"] * 1000) < 1.0
        assert np.allclose(loaded.bounds[0][2], 0.0, atol=1e-4)
    assert not os.path.exists(str(tmp_path / "out.stl.part"))


def test_settings_sheet_mentions_the_essentials(gradient):
    p = with_photos(Params(), gradient)
    build = geometry.build_ornament(p, pitch=PITCH)
    text = export.settings_sheet(p, build.layout, build.stats)
    for word in ("0.12 mm", "Wall loops", "Brim", "Supports", "LED"):
        assert word in text
    assert export.wall_loops_for(p) >= math.ceil(p.t_max / 0.42 / 2)


# ---------------------------------------------------------------------------
# regressions found by review
# ---------------------------------------------------------------------------
def test_abutting_pictures_leave_no_frame_line_between_them(tmp_path):
    """With no gap, the column exactly on the border of two pictures belongs to a
    picture; it used to fall 'outside the photo' by rounding and print as a line."""
    rng = np.random.default_rng(3)
    photo = make_image(tmp_path, "portrait.png", rng.integers(0, 255, (400, 300)))
    for n in (3, 4):
        p = with_photos(Params(n_images=n, projection="wrap", gap_deg=0.0, smoothing=0.0,
                               safe_edges=False), photo)
        lay, _lam, _phis, T = _thickness_grid(p, pitch=0.25)
        plain = np.isclose(T, p.t_frame).all(axis=0)
        assert not plain.any(), f"{int(plain.sum())} plain column(s) with {n} pictures"


def test_transparent_colour_key_png_is_flattened_to_white(tmp_path):
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    path = str(tmp_path / "keyed.png")
    img.save(path, transparency=(0, 0, 0))            # black is marked as see-through
    photo = imaging.load_photo(path)
    assert photo.levels[0].min() > 0.99               # see-through areas come out bright


def test_picture_height_leaves_room_for_the_frame_ramp():
    for kwargs in (dict(relief_side="inside"), dict(relief_side="inside", t_frame=4.0, t_min=0.4),
                   dict(relief_side="outside", t_frame=0.8, t_max=4.0),
                   dict(relief_side="outside", flare_angle=50, cap_diameter=40, t_frame=0.8, t_max=3.2),
                   dict(relief_side="outside", flare_angle=25, t_frame=0.8, t_max=4.0),
                   dict(relief_side="inside", diameter=50, cap_diameter=18, hole_diameter=8, t_frame=3.0)):
        p = Params(band_height_deg=120, **kwargs)
        lo, hi = cap_diameter_range(p)
        p.cap_diameter = min(hi, max(lo, p.cap_diameter))
        lay = derive(p)
        assert lay.notes, "an over-tall band must be reported as limited"
        inside = p.relief_side == "inside"
        step = max(0.0, p.t_frame - p.t_min) if inside else max(0.0, p.t_max - p.t_frame)
        room = (lay.phi_c if inside else lay.phi_t) - lay.band_half       # radians of plain frame
        # the ramp is at its steepest next to the picture, so this is the least room it can need
        needed = step / edge_ramp_slope(lay.band_half) / relief_radius(p)
        assert needed <= room + 1e-9, kwargs
        # ... and it really ends, at full frame thickness, before the shell changes shape
        end = frame_ramp_end(lay.band_half, step, relief_radius(p))
        assert end is not None and end <= (lay.phi_c if inside else lay.phi_t) - math.radians(0.49)
        assert math.degrees(2 * lay.band_half) >= max_band_height_deg(p) - 1e-6
    # with the relief on both sides neither picture edge needs a ramp, so the band can be taller
    assert max_band_height_deg(Params(relief_side="split", t_frame=4.0, t_min=0.4)) > \
        max_band_height_deg(Params(relief_side="inside", t_frame=4.0, t_min=0.4)) + 10


def test_frame_ramp_end_is_where_the_ramp_has_gained_its_height():
    """Add up the ramp numerically: it thickens at tan(70 deg - latitude) all the way."""
    radius, start, step = 37.2, math.radians(50.0), 1.4
    end = frame_ramp_end(start, step, radius)
    lat = np.linspace(start, end, 20001)
    slope = np.tan(math.radians(90.0 - EDGE_RAMP_MIN_DEG) - lat)
    gained = float(np.sum(0.5 * (slope[:-1] + slope[1:]) * np.diff(lat)) * radius)
    assert gained == pytest.approx(step, rel=1e-6)
    assert math.degrees(end) == pytest.approx(57.3, abs=0.1)
    assert frame_ramp_end(start, 3.6, radius) is None        # more than this lean can ever gain


def test_frame_ramp_reaches_frame_thickness_before_the_shell_changes_shape(tmp_path):
    """The worst case from review: tallest band, white photo, thick frame."""
    photo = make_image(tmp_path, "white.png", np.full((100, 100), 255))
    p = Params(relief_side="inside", band_height_deg=120, t_frame=4.0, t_min=0.4, smoothing=0.0)
    for slot in p.images:
        slot.path, slot.auto_levels = photo, False
    lay = derive(p)
    photos, _ = imaging.photos_for(p)
    body = geometry.build_body(p, lay, photos, pitch=0.4, dense=True)
    grid_t = body.thickness[1:body.n_outer].reshape(body.outer_rows, body.n_lon)
    column = int(round(lay.centers[0] / (2 * math.pi / body.n_lon)))
    z = body.vertices[1:body.n_outer].reshape(body.outer_rows, body.n_lon, 3)[:, column, 2]
    lat = np.degrees(np.arcsin(np.clip(z / lay.R, -1, 1)))
    # from the row where the closing cone joins (kept at frame thickness) up to the picture edge
    below = (lat < -math.degrees(lay.band_half)) & (lat > -math.degrees(lay.phi_c) - 0.01)
    t, arc = grid_t[below, column], np.radians(lat[below]) * lay.R
    assert abs(t[0] - p.t_frame) < 1e-6
    growth = -np.diff(t) / np.diff(arc)               # thickening per mm towards the closed end
    assert growth.max() <= edge_ramp_slope(lay.band_half) * 1.05 + 1e-6
    assert t.min() < 1.0                              # ... and it really does start at the thin edge


def test_closed_end_roof_is_reported_and_reasonable():
    assert derive(Params()).roof >= 1.0                               # default: comfortably thick
    thin = derive(Params(t_frame=0.8))
    assert 0.5 <= thin.roof < 0.8
    assert any(c.title.startswith("Thin spot") for c in checks.analyze(Params(t_frame=0.8), thin, 30.0))
    assert not any(c.title.startswith("Thin spot") for c in checks.analyze(Params(), derive(Params()), 30.0))
    assert derive(Params(t_frame=1.4)).roof >= 0.8 - 0.1


def test_width_includes_a_frame_that_stands_proud():
    p = Params(relief_side="outside", t_frame=4.0, t_max=1.4)
    lay = derive(p)
    build = geometry.build_ornament(p, pitch=PITCH)
    widest = float(np.hypot(build.vertices[:, 0], build.vertices[:, 1]).max()) * 2
    assert abs(lay.width - widest) < 0.2              # width is the diameter the part sweeps
    assert lay.width > p.diameter


def test_width_includes_dark_areas_that_stand_proud_with_split_relief(tmp_path):
    black = make_image(tmp_path, "black.png", np.zeros((60, 60)))
    p = Params(relief_side="split", hanger="none", t_frame=1.2, t_max=4.0, gap_deg=0.0)
    for slot in p.images:
        slot.path, slot.auto_levels = black, False
    lay = derive(p)
    build = geometry.build_ornament(p, pitch=PITCH)
    widest = float(np.hypot(build.vertices[:, 0], build.vertices[:, 1]).max()) * 2
    assert abs(lay.width - widest) < 0.2 and lay.width > p.diameter + 1.0
    assert derive(Params(relief_side="split")).width == pytest.approx(80.0, abs=0.9)


def test_wall_loops_cover_the_wall_as_a_layer_cuts_it():
    for p in (Params(), Params(t_max=4.0), Params(t_max=4.0, band_height_deg=60), Params(t_frame=4.0)):
        half = min(math.radians(p.band_height_deg) / 2, band_half_limit(p))
        widest = max(p.t_max, p.t_frame) / math.cos(half)
        assert export.wall_loops_for(p) * 2 * 0.42 >= widest
    assert export.wall_loops_for(Params()) == 6


def test_checklist_names_the_real_shallowest_overhang(gradient):
    p = with_photos(Params(flare_angle=45, cap_diameter=40), gradient)
    text = " ".join(c.detail for c in checks.analyze(p, derive(p), 30.0))
    assert "30 deg from flat" in text and "45 deg from flat" not in text


def test_make_buildable_repairs_every_slider_combination():
    """Whatever the sliders are dragged to, linked values give way until the shape builds."""
    rng = np.random.default_rng(7)
    names = ("diameter", "t_min", "t_max", "t_frame", "hole_diameter", "cap_diameter", "cap_height",
             "cap_wall", "flare_angle", "band_height_deg", "gap_deg", "seat_diameter", "seat_depth")
    hard = [dict(t_frame=4.0, hole_diameter=35.5), dict(t_frame=4.0, diameter=70, hole_diameter=30.5),
            dict(cap_wall=4.0, cap_height=4.0), dict(t_min=1.6, t_max=1.4),
            dict(seat=True, seat_diameter=56.0, seat_depth=30.0),           # far too big a light
            dict(seat=True, seat_diameter=30.6, hole_diameter=30.5),        # no ledge left
            dict(seat=True, seat_diameter=12.0, hole_diameter=45.0, diameter=50.0),
            dict(seat=True, seat_depth=30.0, cap_height=4.0, diameter=150.0, cap_diameter=40.0)]
    seated = 0
    for k in range(600):
        p = Params(relief_side=str(rng.choice(SIDES)))
        if k < len(hard):
            for name, value in hard[k].items():
                setattr(p, name, value)
        else:
            p.seat = bool(k % 2)
            for name in names:
                lo, hi = LIMITS[name]
                setattr(p, name, float(rng.choice([lo, hi, rng.uniform(lo, hi)])))
        asked_for_a_seat = p.seat
        assert make_buildable(p), p
        lay = derive(p)
        for name in names:
            lo, hi = LIMITS[name]
            assert lo - 1e-9 <= getattr(p, name) <= hi + 1e-9, (name, getattr(p, name))
        assert not (p.seat and not asked_for_a_seat)        # a seat never appears by itself
        if p.seat:                                          # a seat that is kept is a real one
            seated += 1
            assert p.hole_diameter <= seat_opening_max(p) + 1e-9
            assert p.cap_diameter >= p.seat_diameter + 2 * MIN_CAP_WALL - 1e-9
            assert lay.z_plate < lay.z_bore < lay.z_seat < lay.z_top and lay.z_ic <= lay.z_plate - 0.5
    assert seated > 60
    untouched = Params()
    before = untouched.to_dict()
    assert make_buildable(untouched) and untouched.to_dict() == before


def test_battery_light_preset_and_the_advice_that_goes_with_it():
    label, size = LIGHT_PRESETS["balloon"][:2]
    assert size == 11.6 and "balloon" in label.lower() and list(LIGHT_PRESETS)[-1] == "custom"
    p = Params(light_preset="balloon", hole_diameter=size)
    lay = derive(p)                                         # fits the standard cap as it is
    assert lay.rh == pytest.approx(5.8) and lay.cup
    sheet = export.settings_sheet(p, lay)
    assert "Push the LED balloon light into the 11.6 mm opening" in sheet
    plain = export.settings_sheet(Params(), derive(Params()))
    assert "Push a tree light through the 10 mm opening" in plain
    for text in (sheet, plain):                             # battery advice goes with every export
        flat = " ".join(text.split())                       # the sheet is wrapped at 78 columns
        assert "Battery lights" in flat and "CR2032" in flat and "bright spot" in flat
        assert "3 x AA" in flat and "timer" in flat and "no batteries" in flat
    fairy = Params(light_preset="fairy", hole_diameter=6.0)
    assert "Feed the fairy lights through the 6 mm opening" in export.settings_sheet(fairy, derive(fairy))


# ---------------------------------------------------------------------------
# a light that sits on the cap (puck seat)
# ---------------------------------------------------------------------------
def _seated(**extra) -> Params:
    """A 30 mm puck's seat on the standard ball, the way the app fits it."""
    settings = dict(seat=True, seat_diameter=30.6, seat_depth=8.0, hole_diameter=25.5, cap_diameter=35.6)
    p = Params(**{**settings, **extra})
    assert make_buildable(p)
    return p


@pytest.mark.parametrize("extra", [
    dict(), dict(hanger="one"), dict(hanger="none"),
    dict(cap_wall=4.0, cap_diameter=33.2),                      # cap wall thicker than the rim
    dict(cap_wall=1.2, cap_diameter=37.0),                      # hollow cap wider than the pocket
    dict(hole_diameter=12.0),                                   # a wide ledge: a tall cone
    dict(seat_diameter=14.0, hole_diameter=11.0, cap_diameter=17.0, cap_wall=3.0),   # tube cap
    dict(diameter=120.0, seat_diameter=38.6, seat_depth=15.0, hole_diameter=33.0, cap_diameter=44.0),
    dict(relief_side="inside"), dict(relief_side="split", nozzle="0.2"),
])
def test_seat_is_a_pocket_with_a_ledge_that_prints_without_support(gradient, extra):
    p = with_photos(_seated(**extra), gradient)
    build = geometry.build_ornament(p, pitch=PITCH)
    lay = build.layout
    report = checks.verify_mesh(build.vertices, build.faces)
    assert report["watertight"] and report["finite"] and report["degenerate_faces"] == 0, report
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    v = build.vertices                              # as printed: the top of the cap is on the bed
    r = np.hypot(v[:, 0], v[:, 1])
    assert abs(v[:, 2].min()) < 1e-6 and abs(v[:, 2].max() - lay.height) < 1e-6

    # the pocket is clear right down to its depth (the hanger loops do not poke into it) ...
    pocket = v[:, 2] < p.seat_depth - 1e-6
    assert r[pocket].min() == pytest.approx(p.seat_diameter / 2, abs=1e-6)
    # ... its floor is a cone that runs down to the opening, which is narrower than the light
    block = lay.z_top - lay.z_plate                 # printer height where the seat ends
    floor = (v[:, 2] > p.seat_depth + 1e-6) & (v[:, 2] < block + 1e-6)
    assert r[floor].min() == pytest.approx(p.hole_diameter / 2, abs=1e-6)
    light = p.seat_diameter - SEAT_CLEARANCE_MM
    assert p.hole_diameter <= light - 2 * SEAT_LEDGE_MIN + SEAT_CLEARANCE_MM + 1e-9
    # the light comes to rest on the cone just below the pocket's floor line, not in the ball
    rest = p.seat_depth + (p.seat_diameter - light) / 2 * math.tan(math.radians(SEAT_CONE_DEG))
    assert p.seat_depth < rest < p.seat_depth + 0.5
    # the bed carries a rim at least as wide as the thinnest cap wall
    assert p.cap_diameter / 2 - p.seat_diameter / 2 >= MIN_CAP_WALL - 1e-9

    # nothing in the cap is a flat ceiling: the ledge is a 45 degree cone
    body = geometry.build_body(p, lay, None, pitch=0.8)
    pv = geometry.to_print_orientation(body.vertices, lay)
    a, b, c = (pv[body.faces[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1)
    centre = (a + b + c) / 3
    down = (area > 1e-9) & (n[:, 2] < -1e-6 * area) & (centre[:, 2] > 1e-6)
    in_seat = down & (centre[:, 2] < block + 1e-6) & (np.hypot(centre[:, 0], centre[:, 1]) < p.seat_diameter / 2)
    assert in_seat.any()
    tilt = np.degrees(np.arccos(np.clip(-n[in_seat, 2] / area[in_seat], -1, 1)))     # 0 = flat ceiling
    assert tilt.min() == pytest.approx(SEAT_CONE_DEG, abs=0.5) and tilt.max() < SEAT_CONE_DEG + 0.5
    bridge = down & (np.abs(centre[:, 2] - (lay.z_top + lay.z_f)) < 1e-6)
    others = down & ~bridge
    every = np.degrees(np.arccos(np.clip(-n[others, 2] / area[others], -1, 1)))
    assert every.min() >= min(p.flare_angle, CLOSURE_ANGLE_DEG) - 0.6


def test_seat_changes_the_cap_and_nothing_else(gradient):
    plain = with_photos(Params(), gradient)
    seated = with_photos(_seated(), gradient)
    a, b = derive(plain), derive(seated)
    assert not a.seat and a.r_seat == 0.0 and a.z_seat == a.z_bore == a.z_top
    assert a.z_plate == pytest.approx(a.z_top - plain.cap_wall)          # the plain top plate
    assert b.seat and b.r_seat == 15.3 and b.rh == 12.75
    assert b.z_top - b.z_seat == pytest.approx(8.0)
    assert b.z_seat - b.z_bore == pytest.approx(15.3 - 12.75)            # 45 degrees
    assert seated.cap_height > plain.cap_height                          # the cap grew for the pocket
    assert seated.cap_height <= 12.0
    # the sphere, the pictures and the closed end are exactly as they were
    for name in ("R", "ro_frame", "ri_frame", "band_half", "panel_half", "phi_c", "r_f", "z_f", "roof"):
        assert getattr(a, name) == getattr(b, name), name
    # the weight shown in the status bar still matches the real mesh
    preview = geometry.build_preview(seated)
    build = geometry.build_ornament(seated, pitch=0.3)
    assert abs(preview.volume_cm3 - build.stats["volume_cm3"]) / build.stats["volume_cm3"] < 0.01
    assert geometry.frame_volume(seated, b) != geometry.frame_volume(plain, a)


def test_seat_gives_way_like_every_other_linked_setting():
    # too wide an opening under the light: the opening gives way, the light keeps its ledge
    p = Params(seat=True, seat_diameter=30.6, hole_diameter=30.0, cap_diameter=35.6, cap_height=12.0)
    with pytest.raises(GeometryError) as err:
        derive(p)
    assert err.value.kind == "window" and "ledge" in str(err.value)
    assert make_buildable(p) and p.seat and p.hole_diameter == 27.5 and p.light_preset == "custom"
    # a cap that is too short for the pocket grows; the pocket keeps its depth
    p = Params(seat=True, seat_diameter=30.6, seat_depth=12.0, hole_diameter=25.5, cap_diameter=35.6)
    with pytest.raises(GeometryError) as err:
        derive(p)
    assert err.value.kind == "cap_short" and "seat" in str(err.value)
    assert make_buildable(p) and p.seat_depth == 12.0 and p.cap_height > 12.0
    # a cap too narrow for the pocket grows as well
    p = Params(seat=True, seat_diameter=30.6, hole_diameter=25.5, cap_diameter=26.0)
    with pytest.raises(GeometryError) as err:
        derive(p)
    assert err.value.kind == "cap_small" and "seat" in str(err.value)
    assert make_buildable(p) and p.cap_diameter == pytest.approx(33.0) and p.seat_diameter == 30.6
    # a seat the sphere cannot hold shrinks, and goes altogether rather than end up useless
    p = Params(seat=True, seat_diameter=50.0, hole_diameter=20.0)
    with pytest.raises(GeometryError) as err:
        derive(p)
    assert err.value.kind == "seat" and "50 mm seat" in str(err.value)
    assert make_buildable(p) and p.seat and p.seat_diameter < 36.0 and p.light_preset == "custom"
    lo, hi = cap_diameter_range(p)
    assert lo <= p.cap_diameter <= hi
    tiny = Params(seat=True, seat_diameter=40.0, diameter=50.0, flare_angle=25.0, t_frame=4.0,
                  hole_diameter=8.0)
    assert make_buildable(tiny)
    derive(tiny)
    # without a seat its sliders mean nothing and never get in the way
    off = Params(seat=False, seat_diameter=56.0, seat_depth=30.0)
    before = off.to_dict()
    assert make_buildable(off) and off.to_dict() == before


def test_checklist_and_guide_describe_the_seat(gradient):
    p = with_photos(_seated(), gradient)
    lay = derive(p)
    results = checks.analyze(p, lay, 33.0, [])
    assert checks.worst_level(results) == checks.OK, [c for c in results if c.level != checks.OK]
    seat = next(c for c in results if c.title == "Seat for the light")
    assert "30.0 mm across" in seat.detail and "8 mm" in seat.detail and "25.5 mm opening" in seat.detail
    bed = next(c for c in results if c.title == "Bed adhesion")
    assert "2.5 mm rim" in bed.detail
    sheet = " ".join(export.settings_sheet(p, lay).split())
    assert "into the seat on top of the cap" in sheet and "lit side down" in sheet
    assert "passes over the light" in sheet
    # one loop hangs the ornament at an angle; that matters with a loose light on top
    p.hanger = "one"
    titles = [c.title for c in checks.analyze(p, derive(p), 33.0, []) if c.level == checks.WARN]
    assert titles == ["One loop tilts the light"]
    # a rim too thin to stand on is pointed out
    thin = with_photos(_seated(cap_diameter=33.2), gradient)
    warned = [c for c in checks.analyze(thin, derive(thin), 33.0, []) if c.level == checks.WARN]
    assert [c.title for c in warned] == ["Thin rim round the seat"] and "1.3 mm rim" in warned[0].detail
    assert "Seat for the light" not in [c.title for c in checks.analyze(Params(), derive(Params()))]


def test_shoulder_for_cap_finds_the_gentlest_angle_that_makes_room():
    p = Params()
    lo, hi = cap_diameter_range(p)
    assert shoulder_for_cap(p, hi) == 30.0 and shoulder_for_cap(p, 20.0) == 30.0    # fits as it is
    angle = shoulder_for_cap(p, 41.5)
    assert angle is not None and angle == round(angle) and 30.0 < angle <= 36.0
    p.flare_angle = angle
    assert cap_diameter_range(p)[1] >= 41.5
    p.flare_angle = angle - 1
    assert cap_diameter_range(p)[1] < 41.5                       # one degree less is not enough
    assert shoulder_for_cap(Params(), 70.0) is None              # wider than the shoulder at any angle
    assert shoulder_for_cap(Params(flare_angle=45), 20.0) == 45.0   # never made gentler
    untouched = Params()
    shoulder_for_cap(untouched, 41.5)
    assert untouched.to_dict() == Params().to_dict()


def test_seat_presets_carry_a_pocket_and_plain_presets_do_not():
    seats = {key: preset for key, preset in LIGHT_PRESETS.items() if preset.seat is not None}
    assert set(seats) == {"puck", "tealight"} and list(LIGHT_PRESETS)[-1] == "custom"
    for key, preset in seats.items():
        assert preset.depth and preset.hole <= preset.seat - 2 * SEAT_LEDGE_MIN, key
        assert LIMITS["seat_diameter"][0] <= preset.seat <= LIMITS["seat_diameter"][1], key
        assert len(preset.label) <= 48, key                  # fits the drop-down in the side panel
    assert LIGHT_PRESETS["mini"].seat is None and LIGHT_PRESETS["custom"].hole is None


# ---------------------------------------------------------------------------
# nozzles
# ---------------------------------------------------------------------------
def test_the_fine_nozzle_is_the_same_design_at_half_the_line_width(gradient):
    coarse, fine = with_photos(Params(), gradient), with_photos(Params(nozzle="0.2"), gradient)
    assert list(NOZZLES) == ["0.4", "0.2"] and coarse.nozzle == "0.4"
    assert (line_width(coarse), line_width(fine)) == (0.42, 0.22)
    assert (recommended_layer_height(coarse), recommended_layer_height(fine)) == (0.12, 0.06)

    # every overhang of the design steps sideways by no bigger a share of a line
    def step(p: Params, angle: float) -> float:
        return recommended_layer_height(p) / math.tan(math.radians(angle)) / line_width(p)

    for angle in (fine.flare_angle, CLOSURE_ANGLE_DEG, OUTSIDE_RAMP_MIN_DEG, EDGE_RAMP_MIN_DEG, SEAT_CONE_DEG):
        assert step(fine, angle) <= step(coarse, angle) + 1e-9, angle
    assert step(fine, fine.flare_angle) < 0.5                    # under half a line on the shoulder

    # the model is sampled finer to match (and the finest setting stays under a line width)
    for quality in QUALITY_PRESETS:
        a, b = Params(quality=quality), Params(quality=quality, nozzle="0.2")
        assert mesh_pitch(b) < mesh_pitch(a)
        assert mesh_pitch(b) == NOZZLES["0.2"].pitches[quality]
    assert (mesh_pitch(coarse), mesh_pitch(fine)) == (0.25, 0.15)
    assert mesh_pitch(fine) < 0.75 * line_width(fine)
    build = geometry.build_ornament(with_photos(Params(nozzle="0.2", quality="draft"), gradient))
    assert build.stats["pitch_mm"] == 0.25

    # walls only, whatever the line width
    for p in (fine, Params(nozzle="0.2", t_max=4.0), Params(nozzle="0.2", t_frame=4.0)):
        half = min(math.radians(p.band_height_deg) / 2, band_half_limit(p))
        widest = max(p.t_max, p.t_frame) / math.cos(half)
        assert export.wall_loops_for(p) * 2 * 0.22 >= widest
    assert (export.wall_loops_for(coarse), export.wall_loops_for(fine)) == (6, 11)

    # Bambu Studio's own presets for the nozzle
    assert export.process_preset(fine) == "0.06mm High Quality @BBL A1 0.2 nozzle"
    assert export.machine_preset(fine) == "Bambu Lab A1 0.2 nozzle"
    assert export.filament_preset(fine) == "Bambu PLA Basic @BBL A1 0.2 nozzle"
    assert export.machine_preset(coarse) == "Bambu Lab A1 0.4 nozzle"
    assert export.filament_preset(coarse) == "Bambu PLA Basic @BBL A1"
    # a shallower shoulder cannot buy a thinner layer than the nozzle has a preset for
    assert recommended_layer_height(Params(nozzle="0.2", flare_angle=25)) == 0.06
    assert recommended_layer_height(Params(flare_angle=25)) == 0.08

    # the price: about three times the print time
    ratio = export.estimated_print_hours(fine, 39.2) / export.estimated_print_hours(coarse, 39.2)
    assert 2.6 < ratio < 3.4 and 13.0 < export.estimated_print_hours(fine, 39.2) < 15.5

    # the checklist and the guide speak of the nozzle that is chosen
    lay = derive(fine)
    results = checks.analyze(fine, lay, 39.2, [])
    assert checks.worst_level(results) == checks.OK, [c for c in results if c.level != checks.OK]
    text = " ".join(c.detail for c in results)
    assert "0.2 mm nozzle" in text and "0.06 mm layers" in text and "0.12 mm" not in text
    sheet = export.settings_sheet(fine, lay)
    assert "0.2 mm nozzle" in sheet and "0.06 mm" in sheet and "0.06mm High Quality @BBL A1 0.2 nozzle" in sheet
    assert "Fit the 0.2 mm nozzle" in " ".join(sheet.split())
    plain = export.settings_sheet(coarse, derive(coarse))
    assert "0.4 mm nozzle" in plain and "0.2 mm nozzle" not in plain

    # a thin highlight is judged against the nozzle that prints it: the wall has to be wide
    # enough for two of its lines side by side even where it is thinnest across
    for t_min, warned in ((0.8, False), (0.76, False), (0.75, True), (0.6, True), (0.4, True)):
        thin = Params(t_min=t_min)
        found = [c for c in checks.analyze(thin, derive(thin)) if c.title == "Very thin highlights"]
        assert bool(found) == warned, t_min
        if found:
            assert found[0].level == checks.WARN and "0.80 mm or more (or the 0.2 mm nozzle)" in found[0].detail
        thin.nozzle = "0.2"                                  # 0.4 mm is still two 0.2 mm lines
        assert not any(c.title == "Very thin highlights" for c in checks.analyze(thin, derive(thin)))


def test_nozzle_is_saved_with_the_project_and_old_files_get_the_standard_one():
    p = Params(nozzle="0.2")
    assert Params.from_json(p.to_json()).nozzle == "0.2"
    assert Params.from_dict({"nozzle": "0.6"}).nozzle == "0.4"          # not a nozzle the app knows
    old = Params().to_dict()
    del old["nozzle"], old["seat"], old["seat_diameter"], old["seat_depth"]
    loaded = params_from_file({"version": "1.3.0", "params": old})
    assert loaded.nozzle == "0.4" and not loaded.seat and loaded.to_dict() == Params().to_dict()
