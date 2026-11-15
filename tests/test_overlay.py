"""Lettering and Christmas decorations laid over the pictures (lithosphere/overlay.py).

    python -m pytest tests/test_overlay.py -q
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithosphere import checks, fonts, geometry, imaging, overlay  # noqa: E402
from lithosphere.params import (DECO_BOTTOM, DECO_FRAME, DECO_SCATTER, DECO_TOP,  # noqa: E402
                                MAX_TEXT_CHARS, ImageSlot, Params, derive, least_true_wall,
                                make_buildable)

RADIUS = 40.0
PANEL = overlay.Panel(RADIUS, math.radians(54.0), math.radians(50.0))      # one of three on 80 mm
GREETING = "Happy Holidays\nfrom\nJohnny and Rachel\n2026"


@pytest.fixture()
def photo(tmp_path) -> str:
    """Left-to-right ramp, dark to bright."""
    path = str(tmp_path / "ramp.png")
    Image.fromarray(np.tile(np.linspace(0, 235, 300), (240, 1)).astype(np.uint8)).save(path)
    return path


def with_photos(p: Params, path: str, slots=None) -> Params:
    for k in range(len(p.images)) if slots is None else slots:
        p.images[k].path = path
    return p


def thickness(p: Params, pitch: float = 0.5):
    """(layout, longitudes, latitudes, wall thickness over the picture band)."""
    lay = derive(p)
    photos, problems = imaging.photos_for(p)
    n_lon = geometry.longitude_count(lay.R, pitch)
    lam = np.arange(n_lon) * (2 * math.pi / n_lon)
    phis, band = geometry._relief_rows(lay, pitch, -math.pi / 2)
    return lay, lam, phis[band], imaging.thickness_band(p, lay, lam, phis[band], photos, pitch), problems


def columns(lay, lam: np.ndarray, k: int) -> np.ndarray:
    """Which longitudes belong to picture k."""
    return np.abs(imaging.wrap_angle(lam - lay.centers[k])) <= lay.panel_half


def coverage(slot: ImageSlot, panel: overlay.Panel = PANEL, step_mm: float = 0.5):
    """(dlam, phi, hidden, light, plain) of a slot's layer on a grid of the picture."""
    layer = overlay.layer_for(slot, panel, 0.3)
    phi = np.arange(-panel.band_half, panel.band_half, step_mm / panel.radius)
    dlam = np.arange(-panel.panel_half, panel.panel_half, step_mm / panel.radius)
    dl, ph = np.meshgrid(dlam, phi)
    if layer is None:                                   # nothing lies over this picture
        return (dl, ph) + (np.zeros(dl.shape),) * 3
    return (dl, ph) + layer.sample(dl, ph)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_lettering_and_decorations_are_saved_with_the_project_and_old_files_get_none():
    p = Params()
    s = p.images[1]
    s.text, s.text_font, s.text_tone, s.text_size = "Happy\nHolidays", "Georgia Bold", "dark", 0.5
    s.text_x, s.text_y, s.text_outline, s.text_bold = 0.3, 0.4, 0.4, 0.2
    s.deco_top, s.deco_bottom, s.deco_frame, s.deco_scatter = "garland", "trees", "candy", "stars"
    s.deco_amount, s.deco_size, s.deco_seed, s.backdrop = 0.9, 1.3, 7, "dark"
    again = Params.from_json(p.to_json())
    assert again.images[1] == s and again.images[1].decorated and not again.images[0].decorated

    old = p.to_dict()                                   # a file written before version 1.5
    for raw in old["images"]:
        for key in [k for k in raw if k.startswith(("text", "deco", "backdrop"))]:
            del raw[key]
    plain = Params.from_dict(old).images[1]
    assert plain == ImageSlot() and not plain.decorated

    bad = p.to_dict()                                   # nonsense is put right, never an error
    bad["images"][0].update(text=None, text_tone="neon", deco_top="tinsel", text_size=9,
                            deco_seed="x", backdrop=3, text_y="up", deco_size=-4, text_bold=99)
    fixed = Params.from_dict(bad).images[0]
    assert (fixed.text, fixed.text_tone, fixed.deco_top, fixed.backdrop) == ("", "light", "none", "plain")
    assert (fixed.text_size, fixed.deco_seed, fixed.text_y) == (1.0, 0, ImageSlot().text_y)
    assert (fixed.deco_size, fixed.text_bold) == (0.6, 1.0)
    huge = p.to_dict()                                  # numbers no setting could ever be
    huge.update(n_images=float("inf"), diameter=10 ** 400)
    huge["images"][0].update(text_size=10 ** 400, deco_amount=-(10 ** 400))
    tamed = Params.from_dict(huge)
    assert (tamed.n_images, tamed.diameter) == (Params().n_images, Params().diameter)
    assert (tamed.images[0].text_size, tamed.images[0].deco_amount) == (ImageSlot().text_size, 0.5)
    for nothing in (None, 7, "x", {"a": 1}):            # no list of pictures at all
        assert Params.from_dict(dict(p.to_dict(), images=nothing)).images == Params().images
    odd = p.to_dict()                                   # not even the right kind of thing
    odd["images"][0].update(text=["a"], deco_frame=["line"], deco_seed=float("inf"),
                            text_size=float("nan"), deco_scatter={"x": 1}, text_font=7)
    cleaned = Params.from_dict(odd).images[0]
    assert (cleaned.deco_frame, cleaned.deco_scatter, cleaned.deco_seed) == ("none", "none", 0)
    assert cleaned.text_size == ImageSlot().text_size and cleaned.text_font == "7"
    assert isinstance(cleaned.text, str)
    long = Params()
    long.images[0].text = "x" * 1000
    long.sanitize()
    assert len(long.images[0].text) == MAX_TEXT_CHARS


def test_the_photo_and_what_lies_over_it_are_reset_separately():
    slot = ImageSlot(path="a.jpg", zoom=2.0, rotation=90, text="Hi", deco_top="lights", backdrop="dark")
    slot.reset_adjustments()                            # the photo's cropping and tones only
    assert (slot.zoom, slot.rotation, slot.path) == (1.0, 0, "a.jpg")
    assert (slot.text, slot.deco_top, slot.backdrop) == ("Hi", "lights", "dark")
    slot.zoom = 3.0
    slot.clear_overlay()                                # ... and the lettering and decorations only
    assert not slot.decorated and (slot.path, slot.zoom, slot.backdrop) == ("a.jpg", 3.0, "plain")
    fresh = ImageSlot(path="b.jpg").keep_overlay_of(ImageSlot(path="a.jpg", zoom=2.0, text="Yo",
                                                              deco_scatter="stars"))
    assert (fresh.path, fresh.zoom, fresh.text, fresh.deco_scatter) == ("b.jpg", 1.0, "Yo", "stars")
    assert not ImageSlot(text="  \n ").decorated and ImageSlot(deco_frame="line").decorated
    for name, choices in (("deco_top", DECO_TOP), ("deco_bottom", DECO_BOTTOM),
                          ("deco_frame", DECO_FRAME), ("deco_scatter", DECO_SCATTER)):
        assert next(iter(choices)) == "none" == getattr(ImageSlot(), name)


def test_fonts_are_found_and_a_missing_one_never_stops_the_lettering():
    have = fonts.installed()
    default = fonts.default_font()
    assert default in have or default == fonts.BUILT_IN
    assert set(fonts.festive()) <= set(have) and fonts.has(default)
    assert not any(word in name.lower() for name in have for word in ("wingdings", "webdings", "marlett"))
    assert "Symbol" not in have                         # Greek letters and signs, not an alphabet
    ghost = "No Such Font Anywhere"
    assert not fonts.has(ghost) and fonts.resolve(ghost) == fonts.resolve(default)
    box = fonts.load(ghost, 40).getbbox("Ag")
    assert box[2] - box[0] > 10 and box[3] - box[1] > 10
    text = overlay.lettering_for(ImageSlot(text="Joy", text_font=ghost), PANEL)
    assert text.substituted and text.font == fonts.resolve(default)
    assert not overlay.lettering_for(ImageSlot(text="Joy"), PANEL).substituted     # "" = the default
    for name in fonts.festive()[:8]:                    # sturdy strokes: that is why they are listed
        assert overlay.stroke_share(name) > 0.08, name
    # a collection file holds several fonts, and not always in the order Windows lists them:
    # each name gets the font that really is called that
    checked = 0
    for name, (path, _listed) in have.items():
        if not path.lower().endswith(".ttc"):
            continue
        from PIL import ImageFont
        names = []
        for face in range(24):
            try:
                family, style = ImageFont.truetype(path, 20, index=face).getname()
            except Exception:
                break
            names.append(f"{family} {style}")
        if name in names:
            family, style = fonts.load(name, 20).getname()
            assert f"{family} {style}" == name, (name, path)
            checked += 1
    assert checked or not any(path.lower().endswith(".ttc") for path, _ in have.values())


# ---------------------------------------------------------------------------
# lettering
# ---------------------------------------------------------------------------
def lay_out(panel: overlay.Panel = PANEL, **settings) -> overlay.Lettering:
    settings.setdefault("text", "Happy Holidays")
    return overlay.lettering_for(ImageSlot(**settings), panel)


def test_text_lines_are_tidied():
    assert overlay.text_lines("\n  a \n\n b\r\n\n") == ["a", "", "b"]
    # a tab or a thin space pasted in is a plain space: fonts have no shape for those
    assert overlay.text_lines("Merry\tChristmas\u2009and\u2003a  happy\u202fNew Year") == [
        "Merry Christmas and a happy New Year"]
    assert overlay.missing_characters(fonts.resolve(fonts.default_font()),
                                      "".join(overlay.text_lines("Merry\tChristmas"))) == ""
    assert overlay.text_lines("1\n2\n3\n4\n5\n6\n7\n8") == ["1", "2", "3", "4", "5", "6"]
    assert overlay.text_lines("") == [] and overlay.text_lines(None) == []
    assert overlay.lettering_for(ImageSlot(text=" \n "), PANEL) is None


def test_lettering_is_as_large_as_its_longest_line_allows():
    full = lay_out(text_size=1.0, text_y=0.0)
    half = lay_out(text_size=0.5, text_y=0.0)
    assert half.size == pytest.approx(full.size / 2, rel=0.01)
    assert half.cap == pytest.approx(full.cap / 2, rel=0.01) and full.cap > 5.0
    one_line = lay_out(text=GREETING.replace("\n", " "), text_size=1.0, text_y=0.0)
    four_lines = lay_out(text=GREETING, text_size=1.0, text_y=0.0)
    assert one_line.cap < 0.5 * full.cap                # a long line makes everything small ...
    assert four_lines.cap > 1.7 * one_line.cap          # ... and breaking it up makes it bigger
    assert four_lines.lines == tuple(GREETING.split("\n"))
    # thickening widens every stroke by twice the setting (and the block makes room for it)
    bold = lay_out(text_size=0.5, text_y=0.0, text_bold=0.3)
    assert bold.stroke - bold.size * half.stroke / half.size == pytest.approx(0.6, abs=0.001)
    assert bold.cap == pytest.approx(half.cap, rel=0.02) and bold.cap < half.cap


@pytest.mark.parametrize("half_deg", [39.0, 54.0, 84.0])
def test_lettering_never_sticks_out_of_its_picture(half_deg):
    """Wherever it is slid to and whatever its size: every corner of the block, edge and all,
    is inside the picture.  Near the top and bottom, where the picture is narrower, the block
    is made smaller instead."""
    panel = overlay.Panel(RADIUS, math.radians(half_deg), math.radians(50.0))
    for size in (0.3, 0.7, 1.0):
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, -0.5, 0.0, 0.5, 1.0):
                text = lay_out(panel, text=GREETING, text_size=size, text_x=x, text_y=y)
                for su in (-1, 1):
                    for sv in (-1, 1):
                        dlam, phi = overlay.offset(0.0, 0.0, text.u + su * text.width / 2,
                                                   text.v + sv * text.height / 2, RADIUS)
                        assert abs(float(dlam)) <= panel.panel_half + 1e-9, (size, x, y)
                        assert abs(float(phi)) <= panel.band_half + 1e-9, (size, x, y)
    middle = lay_out(panel, text_size=1.0, text_y=0.0)
    top = lay_out(panel, text_size=1.0, text_y=1.0)
    assert top.v > 0.5 * RADIUS * panel.band_half and top.size < middle.size
    assert lay_out(panel, text_size=0.4, text_y=1.0).size == pytest.approx(0.4 * middle.size, rel=0.01)
    assert lay_out(panel, text_size=0.4, text_x=1.0, text_y=0.0).u > 5.0
    assert lay_out(panel, text_size=1.0, text_x=1.0, text_y=0.0).u == pytest.approx(0.0, abs=0.05)


def _ink(slot: ImageSlot, panel: overlay.Panel, step_mm: float = 0.2):
    """Where a slot's layer hides the picture, on a grid that runs 3 mm past the sides of
    the picture: (longitude, latitude, hidden) at every point of the grid."""
    layer = overlay.layer_for(slot, panel, 0.2)
    reach = panel.panel_half + 3.0 / panel.radius
    dlam = np.arange(-reach, reach, step_mm / panel.radius)
    phi = np.arange(-panel.band_half, panel.band_half, step_mm / panel.radius)
    dl, ph = np.meshgrid(dlam, phi)
    return dl, ph, layer.sample(dl, ph)[0]


@pytest.mark.parametrize("font", ["Monotype Corsiva", "Lucida Calligraphy Italic", "Bradley Hand ITC",
                                  "Georgia Bold Italic", "Brush Script MT Italic", ""])
def test_the_ink_of_a_leaning_font_is_centred_and_stays_in_its_picture(font):
    """Italics and scripts lean out of the room their font sets aside for each letter.  The
    lettering is centred on what is actually drawn, so at full size it still ends inside the
    picture on both sides, and equally far from each."""
    if font and not fonts.has(font):
        pytest.skip(f"{font} is not on this computer")
    for text in ("Noel", "Joy"):
        slot = ImageSlot(text=text, text_font=font, text_size=1.0, text_y=0.0, text_outline=0.0)
        placed = overlay.lettering_for(slot, PANEL)
        dl, ph, hidden = _ink(slot, PANEL)
        inked = hidden > 0.5
        assert inked.any()
        outside = np.abs(dl) > PANEL.panel_half
        assert not (inked & outside).any(), (font, text)
        # on the flat sheet, where the block was laid out: the ink is in the middle of it
        east = imaging.panel_uv(dl[inked], ph[inked], "natural")[0] * RADIUS
        assert abs(east.max() + east.min()) / 2 < 0.35, (font, text, east.min(), east.max())
        assert placed.width - 1.2 < east.max() - east.min() <= placed.width + 0.4


def test_snow_and_icicles_run_from_one_side_of_the_picture_to_the_other():
    for n_half in (39.0, 54.0, 84.0, 174.0):
        panel = overlay.Panel(RADIUS, math.radians(n_half), math.radians(50.0))
        for settings, low in (({"deco_bottom": "snow"}, True), ({"deco_bottom": "trees"}, True),
                              ({"deco_top": "icicles"}, False)):
            layer = overlay.layer_for(ImageSlot(path="photo.jpg", **settings), panel, 0.3)
            phi = (-1.0 if low else 1.0) * (panel.band_half - 1.0 / RADIUS)      # 1 mm from that edge
            dlam = np.linspace(-1.0, 1.0, 61) * (panel.panel_half - 0.5 / (RADIUS * math.cos(phi)))
            hidden, light, _plain = layer.sample(dlam, np.full_like(dlam, phi))
            assert hidden.min() > 0.95 and light.min() > 0.9, (n_half, settings, hidden.min())


def test_fairy_lights_never_hang_into_each_other():
    """Bulbs are spread over each swag and hang straight down: where two swags meet the wire
    turns a corner, and bulbs hung square to it would lean into each other."""
    for radius, n_half in ((40.0, 54.0), (40.0, 84.0), (40.0, 174.0), (25.0, 54.0), (75.0, 39.0)):
        panel = overlay.Panel(radius, math.radians(n_half), math.radians(50.0))
        s = overlay.deco_unit(ImageSlot(deco_top="lights"), panel)
        spots = overlay.light_spots(panel, s)
        assert len(spots) >= 3
        offset, length, half = (v * s for v in overlay.LIGHT_BODY)
        # two neighbours that lean towards each other come closer by this much at their middles
        lean = 2 * (offset + length / 2) * math.sin(overlay.LIGHT_TILT)
        for (lon_a, lat_a), (lon_b, lat_b) in zip(spots, spots[1:]):
            apart = radius * math.cos((lat_a + lat_b) / 2) * (lon_b - lon_a)
            assert apart - 2 * half - lean > 1.2, (radius, n_half, apart)      # clear dark between them
        assert all(abs(lon) < panel.panel_half and lat < panel.band_half for lon, lat in spots)


def test_lettering_keeps_out_of_the_strips_that_decorations_take_up():
    edge = RADIUS * PANEL.band_half
    plain = lay_out(text=GREETING, text_size=1.0, text_y=0.0)
    for top, bottom, frame in (("garland", "garland", "none"), ("lights", "trees", "none"),
                               ("icicles", "snow", "candy"), ("none", "garland", "line")):
        deep_top, deep_bottom, sides = overlay.reserved(
            ImageSlot(deco_top=top, deco_bottom=bottom, deco_frame=frame), PANEL)
        assert deep_top >= sides and deep_bottom >= sides
        for y in (-1.0, 0.0, 1.0):
            text = lay_out(text=GREETING, text_size=1.0, text_y=y, deco_top=top, deco_bottom=bottom,
                           deco_frame=frame)
            assert text.v + text.height / 2 <= edge - deep_top + 1e-6
            assert text.v - text.height / 2 >= -(edge - deep_bottom) - 1e-6
            assert text.size <= plain.size + 1e-9
    # holly sits in the corners and a wreath round the middle: neither takes a strip
    assert overlay.reserved(ImageSlot(deco_top="holly", deco_bottom="holly", deco_frame="wreath"),
                            PANEL) == (0.0, 0.0, 0.0)
    # decorations that leave almost no room in between do not squeeze the lettering to nothing
    tiny = overlay.Panel(25.0, math.radians(39.0), math.radians(25.0))
    crowded = ImageSlot(text="Joy", deco_top="garland", deco_bottom="garland", deco_size=1.6)
    assert overlay.lettering_for(crowded, tiny).cap > 2.0


def test_a_picture_that_goes_all_the_way_round_takes_lettering_too(photo):
    panel = overlay.Panel(RADIUS, math.radians(174.0), math.radians(50.0))
    assert panel.projection == "wrap" and PANEL.projection == "natural"
    equator = lay_out(panel, text_size=0.3, text_y=0.0)
    high = lay_out(panel, text_size=0.3, text_y=1.0)
    assert equator.stretch == pytest.approx(1.0) and high.stretch > 1.2      # made up for on the sheet
    p = with_photos(Params(n_images=1), photo)
    p.images[0].text, p.images[0].text_y, p.images[0].deco_top = "Joy to the World", 0.0, "lights"
    lay, lam, phis, T, _problems = thickness(p)
    assert T.min() == pytest.approx(p.t_min, abs=0.02) and T.max() == pytest.approx(p.t_max, abs=0.02)
    build = geometry.build_ornament(p, pitch=0.6)
    assert checks.verify_mesh(build.vertices, build.faces)["watertight"]


# ---------------------------------------------------------------------------
# drawing on the sphere
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("half_deg", [54.0, 84.0, 174.0])
def test_shapes_are_true_to_size_wherever_they_sit(half_deg):
    """A 3 mm disc is 3 mm in every direction, measured along the surface, in the middle of
    the picture, in a corner and near the top - whichever way the sheet is laid on the ball."""
    panel = overlay.Panel(RADIUS, math.radians(half_deg), math.radians(50.0))
    sheet = overlay.Sheet(panel, 12.0)
    spots = [(0.0, 0.0), (0.6 * panel.panel_half, 0.7 * panel.band_half),
             (-0.8 * panel.panel_half, -0.8 * panel.band_half), (0.1, 0.85 * panel.band_half)]
    for at in spots:
        sheet.disc(at, 3.0, "light", edge=0.0)
    layer = overlay.Layer.of(sheet, soften=False)
    bearing = np.radians(np.arange(0.0, 360.0, 30.0))
    for at in spots:
        for reach, wanted in ((2.6, 1.0), (3.4, 0.0)):
            dlam, phi = overlay.offset(at[0], at[1], reach * np.cos(bearing), reach * np.sin(bearing),
                                       RADIUS)
            hidden, light, plain = layer.sample(dlam, phi)
            assert np.allclose(hidden, wanted, atol=0.02), (half_deg, at, reach, hidden)
            assert np.allclose(light, wanted, atol=0.02) and np.allclose(plain, 0.0, atol=0.02)


def test_offset_walks_true_distances_on_the_sphere():
    def apart(a, b):                                    # great-circle distance, mm
        (l1, p1), (l2, p2) = a, b
        cos_c = math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(l2 - l1)
        return RADIUS * math.acos(max(-1.0, min(1.0, cos_c)))

    for start in ((0.0, 0.0), (0.7, 0.6), (-1.2, -0.8)):
        for east, north in ((5.0, 0.0), (0.0, -7.0), (3.0, 4.0), (-20.0, 15.0)):
            lon, lat = overlay.offset(start[0], start[1], east, north, RADIUS)
            assert apart(start, (float(lon), float(lat))) == pytest.approx(math.hypot(east, north), abs=1e-9)
    lon, lat = overlay.offset(0.3, 0.2, 0.0, 6.0, RADIUS)       # due north keeps the longitude
    assert float(lon) == pytest.approx(0.3) and float(lat) == pytest.approx(0.2 + 6.0 / RADIUS)


def test_tones_stack_the_way_they_are_drawn():
    sheet = overlay.Sheet(PANEL, 10.0)
    sheet.disc((0.0, 0.0), 6.0, "dark")
    sheet.disc((0.0, 0.0), 3.0, "light")                # light on dark
    sheet.disc((0.3, 0.0), 4.0, "plain")
    sheet.disc((0.3, 0.0), 2.0, "clear")                # a hole through to the photo
    layer = overlay.Layer.of(sheet, soften=False)

    def at(dlam, east=0.0):
        lon, lat = overlay.offset(dlam, 0.0, east, 0.0, RADIUS)
        return tuple(round(float(v[0]), 2) for v in layer.sample(np.array([lon]), np.array([lat])))

    assert at(0.0) == (1.0, 1.0, 0.0) and at(0.0, 4.5) == (1.0, 0.0, 0.0)
    assert at(0.3) == (0.0, 0.0, 0.0) and at(0.3, 3.0) == (1.0, 0.0, 1.0)
    assert at(-0.5) == (0.0, 0.0, 0.0)
    p = Params()
    wall = np.full((1,), 1.7)
    for where, wanted in (((0.0, 0.0), p.t_min), ((0.0, 4.5), p.t_max), ((0.3, 3.0), p.t_frame),
                          ((0.3, 0.0), 1.7), ((-0.5, 0.0), 1.7)):
        lon, lat = overlay.offset(where[0], 0.0, where[1], 0.0, RADIUS)
        assert overlay.apply(wall, layer, np.array([lon]), np.array([lat]), p)[0] == pytest.approx(wanted, abs=0.01)
    sheet.swap = True                                   # on a bright panel the tones change places
    sheet.disc((-0.5, 0.0), 3.0, "light")
    assert tuple(round(float(v[0]), 2) for v in overlay.Layer.of(sheet, soften=False).sample(
        np.array([-0.5]), np.array([0.0]))) == (1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# decorations
# ---------------------------------------------------------------------------
def test_every_decoration_draws_where_it_belongs():
    edge = RADIUS * PANEL.band_half

    def shares(**settings):
        dl, ph, hidden, _light, _plain = coverage(ImageSlot(path="photo.jpg", **settings))
        from_top, from_bottom = edge - ph * RADIUS, edge + ph * RADIUS
        side = (PANEL.panel_half - np.abs(dl)) * RADIUS * np.cos(ph)
        border = np.minimum(side, np.minimum(from_top, from_bottom))
        middle = (np.abs(ph) < 0.3 * PANEL.band_half) & (np.abs(dl) < 0.3 * PANEL.panel_half)
        return {"top": hidden[from_top < 16].mean(), "bottom": hidden[from_bottom < 16].mean(),
                "border": hidden[border < 6].mean(), "middle": hidden[middle].mean(),
                "all": hidden.mean()}

    for key in DECO_TOP:
        got = shares(deco_top=key)
        if key == "none":
            assert got["all"] == 0.0
        else:
            assert got["top"] > 0.06 and got["bottom"] == 0.0 and got["middle"] == 0.0, key
    for key in DECO_BOTTOM:
        got = shares(deco_bottom=key)
        if key != "none":
            assert got["bottom"] > 0.06 and got["top"] == 0.0 and got["middle"] == 0.0, key
    for key in ("line", "candy"):
        got = shares(deco_frame=key)
        assert got["border"] > 0.2 and got["middle"] == 0.0, key
    for key in DECO_SCATTER:
        got = shares(deco_scatter=key)
        if key != "none":                               # over a photo the middle stays clear
            assert 0.01 < got["all"] < 0.25 and got["middle"] == 0.0, key
            dl, ph, hidden, _l, _p = coverage(ImageSlot(deco_scatter=key))      # a plain panel: all over
            assert hidden[(np.abs(ph) < 0.4 * PANEL.band_half) & (np.abs(dl) < 0.4 * PANEL.panel_half)].mean() > 0.01

    # the wreath is a window: the photo shows through it, the corners outside are plain frame
    dl, ph, hidden, light, plain = coverage(ImageSlot(path="photo.jpg", deco_frame="wreath"))
    centre = (np.abs(ph) < 0.3 * PANEL.band_half) & (np.abs(dl) < 0.3 * PANEL.panel_half)
    corner = (np.abs(ph) > 0.9 * PANEL.band_half) & (np.abs(dl) > 0.9 * PANEL.panel_half)
    assert hidden[centre].max() == 0.0 and plain[corner].min() > 0.99
    a, b, bough = overlay.wreath_axes(PANEL, 1.0)
    u, v = imaging.panel_uv(dl, ph, "natural")
    ring = np.abs(np.hypot(u * RADIUS / a, v * RADIUS / b) - 1.0) < 0.05     # along the stem
    assert hidden[ring].mean() > 0.9 and light[ring].mean() > 0.15


def test_decorations_follow_the_size_of_the_picture_within_what_prints():
    slot = ImageSlot(deco_top="garland")
    small = overlay.Panel(25.0, math.radians(54.0), math.radians(50.0))
    large = overlay.Panel(75.0, math.radians(54.0), math.radians(50.0))
    assert overlay.deco_unit(slot, PANEL) == pytest.approx(1.0, abs=0.01)
    assert overlay.deco_unit(slot, small) == 0.75 and overlay.deco_unit(slot, large) == 1.6
    assert overlay.deco_unit(ImageSlot(deco_size=1.5), PANEL) == pytest.approx(1.5, abs=0.02)
    assert overlay.reserved(ImageSlot(deco_top="garland", deco_size=1.5), PANEL)[0] == pytest.approx(
        1.5 * overlay.reserved(slot, PANEL)[0])


def test_scattered_shapes_keep_their_places_and_keep_off_what_is_there():
    slot = ImageSlot(path="photo.jpg", deco_scatter="snowflakes", text="Happy Holidays", text_y=-0.6,
                     deco_top="holly", deco_bottom="garland")
    points = overlay.scatter_points(slot, PANEL)
    assert len(points) >= 4 and points == overlay.scatter_points(slot, PANEL)       # no chance in it
    assert len(overlay.scatter_points(ImageSlot(path="photo.jpg", deco_scatter="snowflakes"), PANEL)) >= 10
    text = overlay.lettering_for(slot, PANEL)
    edge = RADIUS * PANEL.band_half
    for dlam, phi, size, _turn in points:
        u, v = (float(x) * RADIUS for x in imaging.panel_uv(np.array(dlam), np.array(phi), "natural"))
        assert abs(dlam) < PANEL.panel_half and abs(phi) < PANEL.band_half
        assert not (abs(u - text.u) < text.width / 2 + size and abs(v - text.v) < text.height / 2 + size)
        assert edge + phi * RADIUS > overlay.reserved(slot, PANEL)[1]              # off the garland
        assert (u / (0.60 * PANEL.panel_half * RADIUS)) ** 2 + (v / (0.66 * edge)) ** 2 > 1.0
    for i, a in enumerate(points):                      # and off one another
        for b in points[i + 1:]:
            gap = RADIUS * math.acos(min(1.0, math.sin(a[1]) * math.sin(b[1])
                                         + math.cos(a[1]) * math.cos(b[1]) * math.cos(a[0] - b[0])))
            assert gap > a[2] + b[2]
    # the same places whatever the model's resolution: preview and export agree
    coarse, fine = (overlay.layer_for(slot, PANEL, pitch) for pitch in (0.45, 0.15))
    for dlam, phi, _size, _turn in points:
        for layer in (coarse, fine):
            assert layer.sample(np.array([dlam]), np.array([phi]))[1][0] > 0.9      # the hub is light
    # Shuffle deals them afresh; more of them on request; with a wreath they go round it
    shuffled = ImageSlot(**{**slot.__dict__, "deco_seed": slot.deco_seed + 1})
    assert overlay.scatter_points(shuffled, PANEL) != points
    few, many = (len(overlay.scatter_points(ImageSlot(deco_scatter="stars", deco_amount=a), PANEL))
                 for a in (0.0, 1.0))
    assert 3 <= few < 10 < many
    round_it = overlay.scatter_points(ImageSlot(path="photo.jpg", deco_scatter="stars",
                                                deco_frame="wreath"), PANEL)
    a, b, bough = overlay.wreath_axes(PANEL, 1.0)
    assert round_it and all(
        math.hypot(*(float(x) * RADIUS / axis for x, axis in zip(
            imaging.panel_uv(np.array(dlam), np.array(phi), "natural"), (a + bough, b + bough)))) > 1.0
        for dlam, phi, _size, _turn in round_it)
    assert overlay.scatter_points(ImageSlot(text="no shapes asked for"), PANEL) == []


# ---------------------------------------------------------------------------
# on the ornament
# ---------------------------------------------------------------------------
def test_a_picture_without_a_photo_becomes_a_plain_panel_for_its_lettering(photo):
    p = with_photos(Params(n_images=3), photo, slots=(0, 1))
    slot = p.images[2]
    photos, problems = imaging.photos_for(p)
    assert problems == ["Picture 3 has no photo yet."] and photos[2].placeholder
    slot.text, slot.text_size, slot.text_y = "JOY", 1.0, 0.0
    photos, problems = imaging.photos_for(p)
    assert problems == [] and photos[2].plain and not photos[2].placeholder

    lay, lam, phis, T, problems = thickness(p)
    panel = T[:, columns(lay, lam, 2)]
    assert problems == []
    assert np.isclose(panel, p.t_frame, atol=0.01).mean() > 0.5         # the panel itself: plain frame
    assert np.isclose(panel, p.t_min, atol=0.01).mean() > 0.05          # the letters glow
    assert (panel > p.t_max - 0.1).mean() > 0.01                        # with a dark edge round them
    slot.backdrop = "dark"
    panel = thickness(p)[3][:, columns(lay, lam, 2)]
    assert np.isclose(panel, p.t_max, atol=0.01).mean() > 0.6 and np.isclose(panel, p.t_min, atol=0.01).mean() > 0.05
    slot.backdrop, slot.text_tone = "light", "dark"
    panel = thickness(p)[3][:, columns(lay, lam, 2)]
    assert np.isclose(panel, p.t_min, atol=0.01).mean() > 0.6 and np.isclose(panel, p.t_max, atol=0.01).mean() > 0.05
    # on a bright panel a decoration is drawn dark, or it would not show
    slot.text, slot.deco_scatter = "", "stars"
    panel = thickness(p)[3][:, columns(lay, lam, 2)]
    assert (panel > p.t_max - 0.1).mean() > 0.02
    assert checks.worst_level(checks.analyze(p, lay, 30.0, [])) == checks.OK


def test_lettering_changes_only_its_own_picture_and_ignores_the_photo_settings(photo):
    p = with_photos(Params(), photo)
    lay, lam, phis, before, _ = thickness(p)
    slot = p.images[1]
    slot.text, slot.text_size, slot.text_y, slot.deco_top, slot.deco_scatter = "JOY", 1.0, 0.0, "garland", "snow"
    after = thickness(p)[3]
    mine = columns(lay, lam, 1)
    assert np.array_equal(before[:, ~mine], after[:, ~mine]) and not np.array_equal(before[:, mine], after[:, mine])

    # "glowing" stays the thinnest wall whatever is done to the tones of the photo underneath
    layer = overlay.layer_for(slot, overlay.Panel.of(lay), 0.5)
    dl, ph = np.meshgrid(np.sort(imaging.wrap_angle(lam[mine] - lay.centers[1])), phis)
    hidden, light, plain = layer.sample(dl, ph)
    order = np.argsort(imaging.wrap_angle(lam[mine] - lay.centers[1]))
    for changes in ({}, {"invert": True}, {"gamma": 1.7, "smoothing": 2.5}, {"t_min": 0.6, "t_max": 3.2}):
        for name, value in changes.items():
            setattr(p, name, value)
        wall = thickness(p)[3][:, mine][:, order]
        assert np.allclose(wall[light > 0.999], p.t_min, atol=0.005), changes
        assert np.allclose(wall[(hidden > 0.999) & (light < 0.001)], p.t_max, atol=0.005), changes
        assert (light > 0.999).sum() > 200 and ((hidden > 0.999) & (light < 0.001)).sum() > 20


def test_a_decorated_ornament_is_still_a_solid_that_prints(photo):
    """Lettering and decorations are full of sudden edges.  The finished model still has no
    point that starts in mid-air, no face flatter than designed, and a wall that is nowhere
    thinner across than nine tenths of the thinnest wall."""
    from scipy.spatial import cKDTree
    from test_core import _down_faces

    p = with_photos(Params(n_images=4), photo, slots=(0, 1, 2))
    a, b, c, d = p.images
    a.text, a.text_y, a.deco_top, a.deco_bottom = "Happy Holidays", -1.0, "garland", "garland"
    b.text, b.text_y, b.deco_top, b.deco_scatter = "Johnny & Rachel\n2026", 0.9, "lights", "snowflakes"
    c.deco_frame, c.deco_scatter = "wreath", "stars"
    d.text, d.text_y, d.deco_top, d.deco_bottom, d.deco_scatter = GREETING, 0.0, "holly", "trees", "hearts"
    build = geometry.build_ornament(p, pitch=0.3)
    lay = build.layout
    assert build.problems == []
    report = checks.verify_mesh(build.vertices, build.faces)
    assert report["watertight"] and report["finite"] and report["degenerate_faces"] == 0
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    angle, z, _area = _down_faces(build.vertices, build.faces)
    bridge = (np.abs(z - (lay.z_top + lay.z_f)) < 1e-6) & (angle < 1.0)
    assert angle[~((z < 1e-6) | bridge)].min() >= 20.0 - 0.5

    photos, _ = imaging.photos_for(p)
    body = geometry.build_body(p, lay, photos, pitch=0.3, dense=True)
    outer, inner = body.vertices[:body.n_outer], body.vertices[body.n_outer:]
    radius = np.linalg.norm(inner, axis=1)
    in_band = np.abs(np.arcsin(np.clip(inner[:, 2] / radius, -1, 1))) < lay.band_half
    assert cKDTree(outer).query(inner[in_band])[0].min() >= least_true_wall(p) - 0.03


def test_the_glowing_view_shows_that_lettering_high_up_comes_out_soft():
    """Above the middle the outside of the wall cannot carry a sudden edge, so part of every
    letter goes to the inside, which looks blurred when lit.  The lit preview is coloured by
    what the wall LOOKS like, so it must differ from the true thickness there - and not below."""
    def blur_of(y: float) -> float:
        p = Params(n_images=3)
        for slot in p.images:
            slot.text, slot.text_size, slot.text_y = "Merry Christmas", 0.9, y
        lay = derive(p)
        photos, problems = imaging.photos_for(p)
        assert not problems
        body = geometry.build_body(p, lay, photos, pitch=0.4, dense=True)
        shown = np.abs(body.lit[:body.n_outer] - body.thickness[:body.n_outer])
        return float(np.nanmax(shown))

    assert blur_of(-0.9) < 0.05 and blur_of(0.9) > 0.4


def test_print_sheet_lists_what_each_picture_carries(photo):
    from lithosphere import export

    p = with_photos(Params(n_images=4), photo, slots=(0, 1))
    a, b, c, d = p.images
    a.text, a.deco_top, a.deco_bottom = "Happy Holidays", "garland", "garland"
    c.text, c.deco_scatter = "Joy\nto the World", "stars"
    lay = derive(p)
    lines = export.picture_lines(p, lay)
    assert len(lines) == 4
    assert lines[0].startswith("1. ramp.png; lettering \"Happy Holidays\" in ") and "mm tall" in lines[0]
    assert lines[0].endswith("decorations: fir garland with baubles and bows along the top, "
                             "fir garland with baubles and bows along the bottom")
    assert lines[1] == "2. ramp.png"
    assert lines[2].startswith("3. no photo - a plain panel; lettering \"Joy / to the World\"")
    assert lines[2].endswith("decorations: stars scattered over it")
    assert lines[3] == "4. no photo yet"
    sheet = export.settings_sheet(p, lay)
    assert "\nPictures\n" in sheet and "3. no photo - a plain panel" in sheet
    assert sheet.index("Pictures") < sheet.index("Bambu Studio settings")

    # every choice of every list reads as one would say it
    from lithosphere.params import DECO_BOTTOM, DECO_FRAME, DECO_SCATTER, DECO_TOP
    said = []
    for name, choices in (("deco_top", DECO_TOP), ("deco_bottom", DECO_BOTTOM),
                          ("deco_frame", DECO_FRAME), ("deco_scatter", DECO_SCATTER)):
        for key in choices:
            if key == "none":
                continue
            only = Params(n_images=1)
            setattr(only.images[0], name, key)
            line = export.picture_lines(only, derive(only))[0]
            assert line.startswith("1. no photo - a plain panel; decorations: "), line
            said.append(line.split("decorations: ")[1])
    assert said == [
        "fir garland with baubles and bows along the top", "string of fairy lights along the top",
        "snow and icicles along the top", "holly in the top corners",
        "fir garland with baubles and bows along the bottom", "bank of snow along the bottom",
        "snow with fir trees along the bottom", "holly in the bottom corners",
        "line frame all round", "candy-cane border all round", "round wreath",
        "wreath along the edges",
        "snowflakes scattered over it", "falling snow scattered over it",
        "stars scattered over it", "hearts scattered over it"]


# ---------------------------------------------------------------------------
# checklist
# ---------------------------------------------------------------------------
def test_checklist_speaks_up_about_lettering_that_will_not_print_well(photo):
    def found(p):
        lay = derive(p)
        return {c.title: c for c in checks.analyze(p, lay, 30.0, [])
                if "letter" in c.title.lower() or "font" in c.title.lower()}

    p = with_photos(Params(), photo)
    slot = p.images[0]
    assert found(p) == {}                               # no lettering, nothing to say
    slot.text, slot.text_size, slot.text_y = "JOY", 1.0, -0.2
    ok = found(p)["Lettering"]
    assert ok.level == checks.OK and "Picture 1" in ok.detail and "mm tall" in ok.detail

    slot.text = GREETING.replace("\n", " ")             # one long line: far too small
    small = found(p)["Lettering too small to print clearly"]
    assert small.level == checks.WARN and "shorter lines" in small.detail and "0.2 mm nozzle" in small.detail
    text = overlay.lettering_for(slot, overlay.Panel.of(derive(p)))
    assert text.cap < checks.CLEAR_LETTER_LINES * 0.42
    p.nozzle = "0.2"                                    # the fine nozzle prints letters half the size
    assert ("Lettering too small to print clearly" in found(p)) == (text.cap < checks.CLEAR_LETTER_LINES * 0.22)
    assert "0.2 mm nozzle" not in next(iter(found(p).values())).detail
    p.nozzle = "0.4"

    slot.text, slot.text_size, slot.text_y = "Merry Christmas", 0.8, 1.0     # fine, but high up
    soft = found(p)["Lettering high up will look soft"]
    assert soft.level == checks.WARN and "below the middle" in soft.detail
    slot.text_y = -1.0
    assert set(found(p)) == {"Lettering"}
    slot.text_y, p.relief_side = 1.0, "inside"          # there the whole picture is soft anyway
    assert "Lettering high up will look soft" not in found(p)
    p.relief_side = "outside"
    slot.text, slot.text_size, slot.text_y = "JOY", 1.0, 1.0                 # big bold letters survive up there
    assert set(found(p)) == {"Lettering"}

    # a heart in a font of plain letters would print as an empty box: say so
    slot.text = "J \u2665 R"
    lacking = overlay.missing_characters(fonts.resolve(fonts.default_font()), "J\u2665R")
    assert ("Characters the font does not have" in found(p)) == bool(lacking)
    if lacking:
        note = found(p)["Characters the font does not have"]
        assert note.level == checks.WARN and "\u2665" in note.detail and "empty box" in note.detail
    assert overlay.missing_characters(fonts.resolve(fonts.default_font()), "Happy Holidays 2026 &") == ""
    slot.text = "JOY"

    slot.text_font = "No Such Font Anywhere"
    missing = found(p)["Font not found"]
    assert missing.level == checks.WARN and "No Such Font Anywhere" in missing.detail
    assert fonts.resolve(fonts.default_font()) in missing.detail
    # one picture at a time, for the note under the text box
    p.images[2].text = "x" * 60
    only = checks.lettering_checks(p, derive(p), only=2)
    assert len(only) == 1 and "Picture 3" in only[0].detail and only[0].level == checks.WARN
    assert checks.lettering_checks(p, derive(p), only=1) == []


# ---------------------------------------------------------------------------
# frames, wreaths and the candy-cane border (version 1.6)
# ---------------------------------------------------------------------------
def _rim(dl, ph, panel: overlay.Panel = PANEL):
    """mm from each point of a picture to the nearest edge of it."""
    edge = panel.radius * panel.band_half
    side = (panel.panel_half - np.abs(dl)) * panel.radius * np.cos(ph)
    return np.minimum(side, np.minimum(edge - ph * panel.radius, edge + ph * panel.radius))


@pytest.mark.parametrize("kind", ["line", "candy", "wreath", "wreath_square"])
def test_a_frame_leaves_no_sliver_of_the_photo_outside_it(kind):
    """A border runs right along the edge of its picture.  (The first candy-cane border and
    line frame stood a little way in, and a strip of the photo showed outside them.)"""
    for half_deg in (39.0, 54.0, 84.0):
        panel = overlay.Panel(RADIUS, math.radians(half_deg), math.radians(50.0))
        dl, ph, hidden, _light, _plain = coverage(ImageSlot(path="photo.jpg", deco_frame=kind), panel, 0.3)
        rim = _rim(dl, ph, panel)
        assert hidden[(rim > 0.2) & (rim < 1.0)].min() > 0.97, (kind, half_deg)
        middle = (np.abs(ph) < 0.3 * panel.band_half) & (np.abs(dl) < 0.3 * panel.panel_half)
        assert hidden[middle].max() == 0.0, (kind, half_deg)        # and the photo shows inside it


def _runs(flags: np.ndarray, step: float) -> list[float]:
    """Lengths in mm of the runs of True in a row of samples `step` mm apart (whole runs only)."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8)))
    return [float((b - a) * step) for a, b in zip(edges[:-1], edges[1:]) if flags[a + 1]]


def test_the_wreath_is_a_ring_of_fir_with_baubles_and_a_bow():
    """Many thin needles close together on a dark bough - not a few fat leaves a little apart,
    which read as a laurel wreath."""
    slot = ImageSlot(path="photo.jpg", deco_frame="wreath")
    layer = overlay.layer_for(slot, PANEL, 0.15)
    a, b, bough = overlay.wreath_axes(PANEL, 1.0)
    turn = np.linspace(0.0, 2 * math.pi, 3600, endpoint=False)

    def along(grow: float):
        """(hidden, light) round the oval that lies `grow` mm outside the middle line."""
        lon, lat = overlay.offset(0.0, 0.0, (a + grow) * np.cos(turn), (b + grow) * np.sin(turn), RADIUS)
        hidden, light, _plain = layer.sample(lon, lat)
        return hidden, light, lat

    hidden, light, lat = along(0.0)
    assert hidden.min() > 0.97                                   # a closed ring: no gap anywhere
    assert (light < 0.05).mean() > 0.55                          # of dark bough, the baubles apart
    step = 2 * math.pi * (a + b + 5.0) / 2 / len(turn)           # mm between samples, 2.5 mm out
    hidden, light, lat = along(2.5)
    low = (lat < math.radians(5.0)) & (np.abs(turn - 1.5 * math.pi) > 0.35)     # crisp, clear of the bow
    needles = [r for r in _runs((light > 0.5) & low, step) if r < 2.6]
    reach = low.sum() * step
    assert len(needles) / reach > 0.33, len(needles) / reach     # one every 3 mm or closer
    assert 0.6 < np.median(needles) < 1.8                        # ... and thin
    baubles = [r for r in _runs(along(0.0)[1] > 0.5, step) if 2.4 < r < 4.5]
    assert len(baubles) >= 8
    # the bow sits at the foot, and all of it is inside the picture
    foot = overlay.offset(0.0, 0.0, 0.0, -b, RADIUS)
    assert layer.sample(np.array([foot[0]]), np.array([foot[1]]))[1][0] > 0.9
    dlam = np.linspace(-PANEL.panel_half, PANEL.panel_half, 400)
    below = layer.sample(dlam, np.full_like(dlam, -PANEL.band_half - 0.4 / RADIUS))
    assert below[1].max() < 0.05                                 # nothing light beyond the bottom edge


def test_the_wreath_comes_round_or_along_the_edges_of_the_picture():
    shown = {}
    for kind in ("wreath", "wreath_square"):
        dl, ph, hidden, light, plain = coverage(ImageSlot(path="photo.jpg", deco_frame=kind), PANEL, 0.4)
        shown[kind] = float((hidden < 0.01).mean())              # how much of the photo is left
        corner = (np.abs(ph) > 0.93 * PANEL.band_half) & (np.abs(dl) > 0.93 * PANEL.panel_half)
        assert plain[corner].mean() > 0.8, kind                  # (the corners are rounded off)
        assert light.mean() > 0.04, kind
    assert 0.3 < shown["wreath"] < shown["wreath_square"] - 0.08
    # round: plain beyond the ring.  Along the edges: the ring IS the edge
    dl, ph, hidden, light, plain = coverage(ImageSlot(path="photo.jpg", deco_frame="wreath"), PANEL, 0.4)
    assert plain[(np.abs(ph) > 0.8 * PANEL.band_half) & (np.abs(dl) > 0.8 * PANEL.panel_half)].min() > 0.99
    dl, ph, hidden, light, plain = coverage(ImageSlot(path="photo.jpg", deco_frame="wreath_square"), PANEL, 0.4)
    rim = _rim(dl, ph)
    assert (hidden[(rim > 2.0) & (rim < 8.0)] > 0.5).mean() > 0.9


def test_lettering_goes_inside_a_round_wreath():
    a, b, bough = overlay.wreath_axes(PANEL, 1.0)
    for y in (-1.0, -0.6, 0.0, 0.7, 1.0):
        for size in (1.0, 0.7, 0.4):
            slot = ImageSlot(deco_frame="wreath", text="Merry\nChristmas", text_y=y, text_size=size)
            text = overlay.lettering_for(slot, PANEL)
            for su in (-1.0, 1.0):
                for sv in (-1.0, 1.0):                          # every corner of the block
                    u, v = text.u + su * text.width / 2, text.v + sv * text.height / 2
                    assert (u / (a - bough)) ** 2 + (v / (b - bough)) ** 2 <= 1.0 + 1e-6, (y, size)
            # moved to the foot of the ring it keeps its size instead of shrinking to a speck
            assert text.cap == pytest.approx(overlay.lettering_for(
                ImageSlot(deco_frame="wreath", text="Merry\nChristmas", text_y=0.0, text_size=size), PANEL).cap)
    low = overlay.lettering_for(ImageSlot(deco_frame="wreath", text="Merry\nChristmas", text_y=-1.0,
                                          text_size=0.6), PANEL)
    assert low.v < -5.0 and low.cap > 3.0
    # without the wreath the same lettering has the whole picture
    assert overlay.lettering_for(ImageSlot(text="Merry\nChristmas", text_size=1.0), PANEL).cap > \
        overlay.lettering_for(ImageSlot(deco_frame="wreath", text="Merry\nChristmas", text_size=1.0), PANEL).cap


def test_the_candy_cane_border_has_bold_stripes_on_the_slant():
    layer = overlay.layer_for(ImageSlot(path="photo.jpg", deco_frame="candy"), PANEL, 0.15)
    band = overlay.CANDY_BAND
    dlam = np.linspace(-0.55, 0.55, 2400) * PANEL.panel_half       # the straight middle of the bottom side
    step = float(dlam[1] - dlam[0]) * RADIUS * math.cos(PANEL.band_half)

    def row(inset: float):
        phi = -PANEL.band_half + inset / RADIUS
        hidden, light, _plain = layer.sample(dlam, np.full_like(dlam, phi))
        assert hidden.min() > 0.97
        return light < 0.5                                       # where it is dark

    middle = row(overlay.EDGE_MM + band / 2)
    stripes = _runs(middle, step)
    wide = [s for s in stripes if s > 2.0]                       # (measured along the edge: 1.4 x across)
    narrow = [s for s in stripes if s < 1.7]
    assert len(wide) >= 3 and len(narrow) >= 3 and abs(len(wide) - len(narrow)) <= 1
    assert min(narrow) > 0.9                                     # even the narrow ones print
    assert 0.3 < middle.mean() < 0.6                             # about half of it stays light
    # on the slant: 1.5 mm further in, every stripe has moved about 1.5 mm along the edge
    inner, outer = row(overlay.EDGE_MM + band / 2 + 0.75), row(overlay.EDGE_MM + band / 2 - 0.75)
    best = max(range(-260, 261), key=lambda k: float((np.roll(inner, k) == outer).mean()))
    assert 1.0 < abs(best) * step < 2.0, best * step


def test_a_garland_hangs_down_along_either_edge():
    """Along the bottom too the swags sag towards the floor, with their bows at the points
    they hang from."""
    for top in (True, False):
        dlam, phi, along, joints = overlay._edge_path(PANEL, top, inset=5.6, drop=4.6, swag=24.0, ends=5.8)
        assert len(joints) >= 3                                  # at least two swags on this picture
        hung = np.interp(joints, along, phi)
        lowest = [phi[(along > a) & (along < b)].min() for a, b in zip(joints[:-1], joints[1:])]
        assert max(lowest) < hung.min() - 3.0 / RADIUS
        nearest = (PANEL.band_half - hung.max()) if top else (min(lowest) + PANEL.band_half)
        assert nearest * RADIUS == pytest.approx(5.6, abs=0.2)


# ---------------------------------------------------------------------------
# the plain parts of the ball
# ---------------------------------------------------------------------------
ROOMY = dict(n_images=3, band_height_deg=84.0, gap_deg=16.0)     # room for every decoration


def plain_parts(p: Params, pitch: float = 0.4):
    """(layout, longitudes, latitudes, thickness of a plain ball with only what is drawn on
    its plain parts)."""
    lay = derive(p)
    n_lon = geometry.longitude_count(lay.R, pitch)
    lam = np.arange(n_lon) * (2 * math.pi / n_lon)
    phis, _band, _fine = geometry._relief_grid(lay, pitch, -lay.phi_c, overlay.ball_reach(p, lay))
    T = np.full((len(phis), n_lon), p.t_frame)
    overlay.apply_ball(T, phis, lam, p, lay, pitch)
    return lay, lam, phis, T


def test_ball_decorations_are_saved_with_the_project_and_old_files_have_none():
    from lithosphere.params import BALL_DIVIDERS, BALL_POLE, BALL_RING, params_from_file

    p = Params()
    assert not p.ball_decorated
    p.ball_dividers, p.ball_top, p.ball_bottom, p.ball_pole = "candy", "zigzag", "fir", "star"
    q = Params.from_json(p.to_json())
    assert (q.ball_dividers, q.ball_top, q.ball_bottom, q.ball_pole) == ("candy", "zigzag", "fir", "star")
    assert q.ball_decorated
    old = Params().to_dict()
    for name in ("ball_dividers", "ball_top", "ball_bottom", "ball_pole"):
        del old[name]
    assert not params_from_file({"version": "1.5.0", "params": old}).ball_decorated
    odd = dict(p.to_dict(), ball_dividers="tinsel", ball_top=None, ball_bottom=7, ball_pole=["star"])
    cleaned = Params.from_dict(odd)
    assert (cleaned.ball_dividers, cleaned.ball_top, cleaned.ball_bottom, cleaned.ball_pole) == ("none",) * 4
    assert set(BALL_DIVIDERS) >= {"none", "stars", "candy", "fir"} and "zigzag" in BALL_RING
    assert set(BALL_POLE) == {"none", "snowflake", "star"}


def test_the_rows_of_the_model_are_as_before_unless_the_ball_is_decorated():
    """An undecorated ball is built exactly as it always was; a ring of decoration carries the
    close rows of the pictures on over the strip it is on."""
    for settings in ({}, {"band_height_deg": 60.0}, {"diameter": 120.0, "n_images": 2}):
        p = Params(**settings)
        lay = derive(p)
        for pitch, start in ((0.25, -lay.phi_c), (0.6, -math.pi / 2)):
            phis, band = geometry._relief_rows(lay, pitch, start)
            b = lay.band_half                                   # the rule of version 1.5, spelled out
            nb = max(8, int(round(2 * b * lay.R / pitch)))
            dphi = 2 * b / nb
            rows = -b + (np.arange(nb) + 0.5) * dphi
            lower = np.linspace(start, -b - 0.5 * dphi, max(1, math.ceil((-b - 0.5 * dphi - start)
                                                                         / geometry.COARSE_STEP)) + 1)
            if start <= -math.pi / 2 + 1e-9:
                lower = lower[1:]
            upper = np.linspace(b + 0.5 * dphi, lay.phi_t,
                                max(1, math.ceil((lay.phi_t - b - 0.5 * dphi) / geometry.COARSE_STEP)) + 1)
            assert np.allclose(phis, np.concatenate([lower, rows, upper]), atol=1e-12)
            assert (band.start, band.stop) == (len(lower), len(lower) + nb)
    p = Params(ball_top="stars", ball_bottom="zigzag")
    lay = derive(p)
    phis, band, fine = geometry._relief_grid(lay, 0.25, -lay.phi_c, overlay.ball_reach(p, lay))
    assert fine.start < band.start and fine.stop > band.stop
    assert np.allclose(np.diff(phis[fine]), np.diff(phis[band])[0])                # evenly, like the band
    assert phis[fine][0] < -lay.phi_c + 3 * 0.25 / lay.R + 1e-9 and phis[fine][-1] > lay.phi_t - 3 * 0.25 / lay.R
    assert phis[0] == pytest.approx(-lay.phi_c) and phis[-1] == pytest.approx(lay.phi_t)
    assert np.all(np.diff(phis) > 0)
    plain, plain_band = geometry._relief_rows(lay, 0.25, -lay.phi_c)
    assert np.allclose(phis[band], plain[plain_band])                              # the pictures' own rows stay


@pytest.mark.parametrize("kind", ["stars", "snowflakes", "beads", "candy", "fir"])
def test_dividers_are_decorated_all_but_the_one_at_the_back(kind):
    from lithosphere.params import BACK_LON

    p = Params(ball_dividers=kind, **ROOMY)
    lay, lam, phis, T = plain_parts(p)
    assert overlay.ball_for(p, lay, 0.4).no_room == []
    changed = np.abs(T - p.t_frame) > 0.05
    in_band = np.abs(phis) < lay.band_half
    assert not changed[~in_band].any()                                              # nothing above or below
    for center in lay.centers:                                                      # no picture is touched
        assert not changed[:, np.abs(imaging.wrap_angle(lam - center)) <= lay.panel_half].any()
    centres = overlay.divider_centres(lay, back=True)
    assert len(centres) == 3 and len(overlay.divider_centres(lay)) == 2
    half = overlay.divider_half(lay)
    for centre in centres:
        cols = np.abs(imaging.wrap_angle(lam - centre)) < half
        back = abs(imaging.wrap_angle(centre - BACK_LON)) < 1e-6
        if back:
            assert not changed[:, cols].any()                   # the seam runs here: plain
            continue
        part = T[np.ix_(in_band, cols)]
        assert 0.08 < (np.abs(part - p.t_frame) > 0.01).mean() < 0.95, kind
        assert np.isclose(part, p.t_min, atol=0.02).mean() > 0.03                   # it glows
        if kind in ("candy", "fir"):                            # dark stripes, a dark bough
            assert (part > p.t_frame + 0.3).mean() > 0.02
        else:                                                   # plain glowing shapes: no raised edge
            assert part.max() < p.t_frame + 0.01
        # it keeps clear of the pictures either side, and runs most of the height
        dlam = imaging.wrap_angle(lam[cols] - centre)
        near_edge = (half - np.abs(dlam)) * lay.R < 0.5
        assert not changed[np.ix_(in_band, cols)][:, near_edge].any()
        rows = np.flatnonzero(changed[np.ix_(in_band, cols)].any(axis=1))
        assert (phis[in_band][rows[-1]] - phis[in_band][rows[0]]) > 1.5 * lay.band_half


@pytest.mark.parametrize("kind", ["stars", "snowflakes", "beads", "zigzag", "fir"])
def test_a_ring_goes_round_the_ball_and_leaves_the_back_clear(kind):
    from lithosphere.params import BACK_LON

    p = Params(ball_top=kind, ball_bottom=kind, **ROOMY)
    lay, lam, phis, T = plain_parts(p)
    assert overlay.ball_for(p, lay, 0.4).no_room == []
    changed = np.abs(T - p.t_frame) > 0.05
    assert not changed[np.abs(phis) <= lay.band_half].any()                         # not in the picture band
    above, below, _closed = overlay.ball_strips(lay)
    for lo, hi in (above, below):
        rows = (phis > lo) & (phis < hi)
        strip = changed[rows]
        assert strip.any(), kind
        # clear of both edges of its strip
        edge = np.minimum(phis[rows] - lo, hi - phis[rows]) * lay.R
        assert not strip[edge < 0.5].any()
        # all the way round: something in every 30 degrees of longitude but the back
        off = np.degrees(imaging.wrap_angle(lam - BACK_LON))
        for start in range(-165, 165, 30):
            if abs(start + 15) > 20:
                assert strip[:, (off >= start) & (off < start + 30)].any(), (kind, start)
        mid = 0.5 * (lo + hi)
        clear = np.abs(np.radians(off)) * lay.R * math.cos(mid) < overlay.RING_BACK_GAP - 0.4
        assert not strip[:, clear].any(), kind                                      # the seam's own strip
        light = np.isclose(T[rows], p.t_min, atol=0.02)
        assert light.mean() > 0.02


def test_what_is_round_the_top_is_formed_on_the_inside_of_the_wall(photo):
    """Above the pictures the outside leans over too far to carry a pattern - and would blur
    the top of the pictures trying.  So the outside stays exactly as it is without the ring."""
    plain = with_photos(Params(n_images=3), photo)
    stars = with_photos(Params(n_images=3, ball_top="stars", ball_bottom="stars"), photo)
    beads = with_photos(Params(n_images=3, ball_top="beads", ball_bottom="stars"), photo)
    lay = derive(plain)
    photos, _ = imaging.photos_for(plain)
    a, b, c = (geometry.build_body(p, lay, photos, pitch=0.4, dense=True) for p in (plain, stars, beads))

    def outside(body):
        v = body.vertices[1:body.n_outer]
        rho = np.linalg.norm(v, axis=1)
        return np.arcsin(np.clip(v[:, 2] / rho, -1, 1)), rho

    lat_a, rho_a = outside(a)
    lat_b, rho_b = outside(b)
    lat_c, rho_c = outside(c)
    above, below, _closed = overlay.ball_strips(lay)
    top = (lat_b > above[0]) & (lat_b < above[1])
    bottom = (lat_b > below[0]) & (lat_b < below[1])
    # whatever is drawn round the top, the outside of the ball is the same
    assert np.array_equal(lat_b, lat_c) and np.allclose(rho_b, rho_c, atol=1e-9)
    assert np.abs(rho_b[top] - lay.ro_frame).max() < 0.1        # (the run-in to a dark picture edge)
    assert (rho_b[bottom] < lay.ro_frame - 1.0).mean() > 0.02   # below, it is cut into the outside
    # the wall really is thinner where the stars are - the inside has moved - and not where
    # the beads are
    thick_b, thick_c = b.thickness[1:b.n_outer], c.thickness[1:c.n_outer]
    assert (thick_b[top] < plain.t_min + 0.05).mean() > 0.02
    assert (np.abs(thick_b[top] - thick_c[top]) > 0.5).mean() > 0.02
    # lit, what is formed on the inside looks softer than what is cut into the outside
    soft = np.abs(b.lit[1:b.n_outer] - thick_b)
    assert soft[top].max() > 0.3 and soft[bottom].max() < 0.05
    # and the pictures are built as they are without the rings
    band_a, band_b = np.abs(lat_a) < lay.band_half, np.abs(lat_b) < lay.band_half
    assert np.allclose(np.sort(rho_a[band_a]), np.sort(rho_b[band_b]), atol=0.02)


@pytest.mark.parametrize("kind", ["snowflake", "star"])
def test_the_shape_underneath_only_ever_adds_plastic(kind, photo):
    from lithosphere.params import BACK_LON

    plain = with_photos(Params(n_images=3), photo)
    shaped = with_photos(Params(n_images=3, ball_pole=kind), photo)
    lay = derive(plain)
    photos, _ = imaging.photos_for(plain)
    a = geometry.build_body(plain, lay, photos, pitch=0.3, dense=True)
    b = geometry.build_body(shaped, lay, photos, pitch=0.3, dense=True)
    assert np.array_equal(a.vertices[a.n_outer:], b.vertices[b.n_outer:])           # the inside is untouched
    v = b.vertices[:b.n_outer]
    rho = np.linalg.norm(v, axis=1)
    lat = np.arcsin(np.clip(v[:, 2] / rho, -1, 1))
    cap = lat < -lay.phi_c - 1e-9
    rise = rho[cap] - lay.ro_frame
    assert rise.min() > -1e-9 and rise.max() == pytest.approx(overlay.POLE_RAISE, abs=0.02)
    assert 0.08 < (rise > 0.5 * overlay.POLE_RAISE).mean() < 0.6
    assert rho[0] == pytest.approx(lay.ro_frame + overlay.POLE_RAISE, abs=0.02)    # the middle of it
    # it stops short of where the wall changes shape, and no arm of a snowflake lies on the seam
    arc = (lat[cap] + math.pi / 2) * lay.R
    assert not (rise[arc > (math.pi / 2 - lay.phi_c) * lay.R - overlay.POLE_EDGE + 0.5] > 0.02).any()
    if kind == "snowflake":
        lon = np.arctan2(v[cap, 1], v[cap, 0])
        at_back = (np.abs(imaging.wrap_angle(lon - BACK_LON)) * np.sin(arc / lay.R) * lay.R < 0.5) & (arc > 6.0)
        assert at_back.sum() > 20 and rise[at_back].max() < 0.05
    assert b.picture_mm3 > a.picture_mm3 + 50.0                                     # and it is counted
    # nothing changes anywhere else
    rest_a = np.linalg.norm(a.vertices[:a.n_outer], axis=1)
    lat_a = np.arcsin(np.clip(a.vertices[:a.n_outer, 2] / rest_a, -1, 1))
    assert np.allclose(np.sort(rest_a[lat_a > -lay.phi_c + 1e-9]), np.sort(rho[lat > -lay.phi_c + 1e-9]))


def test_a_decoration_that_finds_no_room_is_left_out_and_the_checklist_says_what_to_change():
    p = Params(n_images=3, ball_top="fir", ball_bottom="zigzag")
    lay = derive(p)
    ball = overlay.ball_for(p, lay, 0.5)
    assert ball.no_room == ["ball_top"] and ball.top is None and ball.bottom is not None
    found = {c.title: c for c in checks.ball_checks(p, lay)}
    need = found["No room round the top"]
    assert need.level == checks.WARN and "7.0 mm" in need.detail and "7.8 mm" in need.detail
    advised = int(need.detail.split("Picture height on the Sphere page to ")[1].split(" ")[0])
    assert "Zigzag round the bottom" in found["Decorations on the ball"].detail
    p.band_height_deg = float(advised)                           # do as it says: now it fits
    assert overlay.ball_for(p, derive(p), 0.5).no_room == []
    assert [c.level for c in checks.ball_checks(p, derive(p))] == [checks.OK]
    p.band_height_deg = float(advised + 2)
    assert overlay.ball_for(p, derive(p), 0.5).no_room == ["ball_top"]

    narrow = Params(n_images=3, gap_deg=6.0, ball_dividers="stars")
    found = {c.title: c for c in checks.ball_checks(narrow, derive(narrow))}
    need = found["No room between the pictures"]
    advised = float(need.detail.split("Gap between pictures on the Sphere page to ")[1].split(" ")[0])
    assert 6.0 < advised <= 12.0 and "stars" in need.detail
    narrow.gap_deg = advised
    assert overlay.ball_for(narrow, derive(narrow), 0.5).no_room == []

    one = Params(n_images=1, ball_dividers="candy", ball_pole="star")
    found = {c.title: c for c in checks.ball_checks(one, derive(one))}
    assert "only gap is the one at the back" in found["Nothing between the pictures to decorate"].detail
    assert "Big star underneath" in found["Decorations on the ball"].detail
    assert checks.ball_checks(Params(), derive(Params())) == []
    # the rest of the checklist carries them
    titles = [c.title for c in checks.analyze(p, derive(p), 30.0, [])]
    assert "No room round the top" in titles


def test_the_checklist_says_what_is_on_the_ball_and_how_it_will_look():
    p = Params(n_images=3, ball_dividers="stars", ball_top="beads", ball_bottom="snowflakes",
               ball_pole="snowflake")
    (only,) = checks.ball_checks(p, derive(p))
    assert only.level == checks.OK and only.title == "Decorations on the ball"
    assert only.detail.startswith("Stars between the pictures, string of beads round the top, row of "
                                  "snowflakes round the bottom, big snowflake underneath.")
    assert "divider at the back stays plain" in only.detail
    assert "inside of the wall" in only.detail and "not at all unlit" in only.detail
    assert "0.6 mm proud" in only.detail


def test_print_sheet_says_what_is_on_the_ball(photo):
    from lithosphere import export

    p = with_photos(Params(n_images=3), photo)
    lay = derive(p)
    assert export.ball_line(p, lay) == "" and "The ball itself" not in export.settings_sheet(p, lay)
    p.ball_dividers, p.ball_top, p.ball_pole = "candy", "fir", "star"
    line = export.ball_line(p, lay)
    assert line == ("The ball itself: candy-cane stripes between the pictures, big star underneath "
                    "(no room for: fir garland round the top)")
    sheet = export.settings_sheet(p, lay)
    assert sheet.index("Pictures") < sheet.index("The ball itself") < sheet.index("Bambu Studio settings")


@pytest.mark.parametrize("side", ["outside", "split", "inside"])
def test_a_ball_decorated_all_over_is_still_a_solid_that_prints(photo, side):
    """Every plain part decorated, the new frames on the pictures: no point starts in mid-air,
    no face is flatter than designed, the wall is nowhere too thin, whichever side carries the
    relief."""
    from scipy.spatial import cKDTree
    from test_core import _down_faces

    p = with_photos(Params(n_images=3, relief_side=side, ball_dividers="stars", ball_top="zigzag",
                           ball_bottom="snowflakes", ball_pole="snowflake"), photo)
    a, b, c = p.images[:3]
    a.deco_frame, b.deco_frame, c.deco_frame = "wreath", "wreath_square", "candy"
    build = geometry.build_ornament(p, pitch=0.3)
    lay = build.layout
    # (with the relief inside there is no ring round the bottom: see the test further down)
    assert build.problems == []
    assert overlay.ball_for(p, lay, 0.3).no_room == (["ball_bottom"] if side == "inside" else [])
    report = checks.verify_mesh(build.vertices, build.faces)
    assert report["watertight"] and report["finite"] and report["degenerate_faces"] == 0
    assert len(checks.hanging_tips(build.vertices, build.faces)) == 0
    angle, z, _area = _down_faces(build.vertices, build.faces)
    bridge = (np.abs(z - (lay.z_top + lay.z_f)) < 1e-6) & (angle < 1.0)
    assert angle[~((z < 1e-6) | bridge)].min() >= 20.0 - 0.5
    if side == "outside":
        photos, _ = imaging.photos_for(p)
        body = geometry.build_body(p, lay, photos, pitch=0.3, dense=True)
        outer, inner = body.vertices[:body.n_outer], body.vertices[body.n_outer:]
        radius = np.linalg.norm(inner, axis=1)
        lat = np.arcsin(np.clip(inner[:, 2] / radius, -1, 1))
        relief = (lat > -lay.phi_c + 0.02) & (lat < lay.phi_t - 0.02)
        assert cKDTree(outer).query(inner[relief])[0].min() >= least_true_wall(p) - 0.03
    # the checklist has nothing to complain about on account of the decorations
    said = [c.title for c in checks.analyze(p, lay, build.stats["volume_cm3"], [])
            if c.level != checks.OK and any(word in c.title.lower() for word in ("ball", "room", "ring"))]
    assert said == (["No ring round the bottom with the picture on the inside"] if side == "inside" else [])


def test_the_checklist_warns_when_the_frame_is_too_thin_for_the_ball_decorations_to_show():
    p = Params(n_images=3, ball_dividers="stars", ball_pole="star", t_frame=1.0)
    found = {c.title: c for c in checks.ball_checks(p, derive(p))}
    faint = found["Decorations on the ball will hardly show"]
    assert faint.level == checks.WARN and "1.30 mm" in faint.detail
    p.t_frame = 1.3
    assert "Decorations on the ball will hardly show" not in {c.title for c in checks.ball_checks(p, derive(p))}
    # the shape underneath is raised, not cut in: it shows whatever the frame is
    only_pole = Params(n_images=3, ball_pole="star", t_frame=1.0)
    assert [c.level for c in checks.ball_checks(only_pole, derive(only_pole))] == [checks.OK]


def test_the_print_time_estimate_counts_the_decorations():
    from lithosphere import export

    base = export.estimated_print_hours(Params(n_images=3), 38.0)
    framed = Params(n_images=3)
    framed.images[0].deco_frame = "wreath"
    lined = Params(n_images=3)
    lined.images[0].deco_frame = "line"
    assert export.estimated_print_hours(framed, 38.0) > export.estimated_print_hours(lined, 38.0) > base
    ball = Params(n_images=3, ball_dividers="stars", ball_top="beads", ball_bottom="snowflakes",
                  ball_pole="star")
    assert base + 0.3 < export.estimated_print_hours(ball, 38.0) < base + 0.6
    # a decoration that finds no room is not printed, and costs nothing
    left_out = Params(n_images=3, ball_top="fir")
    assert export.estimated_print_hours(left_out, 38.0) == pytest.approx(base)


# ---------------------------------------------------------------------------
# what an independent review of version 1.6 found
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extra", [{"deco_top": "garland"}, {"deco_top": "lights"}, {"deco_top": "icicles"},
                                   {"deco_bottom": "garland"}, {"deco_bottom": "trees"},
                                   {"deco_top": "garland", "deco_bottom": "snow"}])
def test_lettering_in_a_wreath_keeps_its_size_when_an_edge_is_decorated_too(extra):
    """The room for the lettering is then off-centre in the ring.  Slid towards the ring it
    used to be squeezed - to a third of a millimetre at the far end - and poked into the bough."""
    a, b, bough = overlay.wreath_axes(PANEL, 1.0)
    sizes = []
    for y in (-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0):
        slot = ImageSlot(deco_frame="wreath", text="Merry\nChristmas", text_y=y, **extra)
        text = overlay.lettering_for(slot, PANEL)
        sizes.append(text.cap)
        top, bottom, _sides = overlay.reserved(slot, PANEL)
        edge = RADIUS * PANEL.band_half - overlay.MARGIN_MM
        assert text.v + text.height / 2 <= edge - top + 1e-6 and text.v - text.height / 2 >= bottom - edge - 1e-6
        for su in (-1.0, 1.0):
            for sv in (-1.0, 1.0):
                u, v = text.u + su * text.width / 2, text.v + sv * text.height / 2
                assert (u / (a - bough)) ** 2 + (v / (b - bough)) ** 2 <= 1.0 + 1e-6, (extra, y)
    assert max(sizes) == pytest.approx(min(sizes)) and min(sizes) > 3.0, sizes


def test_a_band_never_folds_over_itself_and_leaves_a_hole():
    """Bands are laid piece by piece.  Drawn as one outline they folded over where the path
    turned sharply and the fill left a hole: islands in the corners of a zigzag, a fleck of
    photo under the bow where two swags of a garland meet."""
    from scipy import ndimage

    sheet = overlay.Sheet(PANEL, 12.0)
    x = np.linspace(-20.0, 20.0, 400)
    zig = 6.0 * np.abs(((x / 5.0) % 2.0) - 1.0) - 3.0                  # sharp corners every 5 mm
    lon, lat = overlay.offset(0.0, 0.0, x, zig, RADIUS)
    sheet.ribbon(lon, lat, 5.0, "light")                              # wider than the corners are apart
    mask = np.asarray(sheet.image.getchannel(1)) > 127
    assert mask.sum() > 20000
    assert np.array_equal(ndimage.binary_fill_holes(mask), mask)      # no hole anywhere in it
    for corner in range(-20, 21, 5):                                  # and every corner is covered
        cx, cy = sheet.to_px(*overlay.offset(0.0, 0.0, float(corner), float(zig[np.argmin(np.abs(x - corner))]),
                                             RADIUS))
        assert mask[int(round(float(cy))), int(round(float(cx)))]
    # it is as wide as asked on the straight, and rounded where it ends
    cx, cy = sheet.to_px(*overlay.offset(0.0, 0.0, -17.5, 0.0, RADIUS))
    column = mask[:, int(round(float(cx)))]
    assert column.sum() / sheet.px_per_mm == pytest.approx(5.0 / math.cos(math.atan(6.0 / 5.0)), abs=0.5)

    # the garland: solid under every bow, where two swags meet
    layer = overlay.layer_for(ImageSlot(path="photo.jpg", deco_bottom="garland"), PANEL, 0.15)
    dlam, phi, along, joints = overlay._edge_path(PANEL, False, inset=5.6, drop=4.6, swag=24.0, ends=5.8)
    for where in joints[1:-1]:
        at = np.interp(where, along, dlam), np.interp(where, along, phi)
        ring = np.linspace(0.0, 2 * math.pi, 24, endpoint=False)
        lon, lat = overlay.offset(at[0], at[1], 2.6 * np.cos(ring), 2.6 * np.sin(ring), RADIUS)
        assert layer.sample(lon, lat)[0].min() > 0.97                 # nothing of the photo shows


@pytest.mark.parametrize("band", [100.0, 60.0, 40.0])
def test_a_zigzag_keeps_the_gap_at_the_back_where_the_meridians_close_in(band):
    from lithosphere.params import BACK_LON

    for diameter in (80.0, 150.0):
        p = Params(n_images=3, diameter=diameter, band_height_deg=band, ball_top="zigzag", ball_bottom="zigzag")
        assert make_buildable(p)
        lay, lam, phis, T = plain_parts(p, pitch=0.25)
        cut = T < p.t_frame - 0.7                                       # the line itself
        assert cut.any()
        off = np.abs(imaging.wrap_angle(lam - BACK_LON))[None, :] * lay.R * np.cos(phis)[:, None]
        assert off[cut].min() > overlay.RING_BACK_GAP - 0.25, (band, diameter, off[cut].min())
        # and no island of thick wall stands in the groove
        from scipy import ndimage
        for rows in ((phis > lay.band_half), (phis < -lay.band_half)):
            holes = ndimage.binary_fill_holes(np.pad(cut[rows], ((1, 1), (0, 0)))) & ~np.pad(cut[rows], ((1, 1), (0, 0)))
            assert not holes.any(), (band, diameter)


@pytest.mark.parametrize("kind", ["candy", "fir", "stars"])
def test_divider_art_keeps_clear_of_the_pictures_however_wide_the_divider(kind):
    for diameter, gap, height in ((80.0, 12.0, 100.0), (80.0, 20.0, 100.0), (80.0, 30.0, 110.0),
                                  (50.0, 40.0, 100.0), (150.0, 30.0, 110.0)):
        p = Params(n_images=3, diameter=diameter, gap_deg=gap, band_height_deg=height, ball_dividers=kind)
        assert make_buildable(p)                                        # (a small ball needs a smaller cap)
        lay, lam, phis, T = plain_parts(p, pitch=0.25)
        if overlay.ball_for(p, lay, 0.25).no_room:
            continue
        half = overlay.divider_half(lay)
        nearest = []
        for centre in overlay.divider_centres(lay):
            dlam = imaging.wrap_angle(lam - centre)
            cols = np.abs(dlam) < half
            part = np.abs(T[:, cols] - p.t_frame) > 0.5                 # well inside the art, not its soft edge
            assert part.any()
            to_edge = (half - np.abs(dlam[cols]))[None, :] * lay.R * np.cos(phis)[:, None]
            nearest.append(to_edge[part].min())
        assert min(nearest) > overlay.BALL_EDGE - 0.2, (kind, diameter, gap, min(nearest))


def test_a_border_never_swallows_a_narrow_or_low_picture():
    """On the narrow pictures of a ball with four, or on low ones, a wreath at full size left a
    slit of photo, folded over itself in the corners and pushed its bow off the edge."""
    for n, diameter, band, gap, size in ((4, 80.0, 100.0, 12.0, 1.6), (4, 80.0, 100.0, 28.0, 1.6),
                                         (4, 50.0, 100.0, 40.0, 1.0), (3, 80.0, 40.0, 12.0, 1.0),
                                         (2, 50.0, 40.0, 12.0, 1.6), (4, 100.0, 120.0, 20.0, 1.6)):
        p = Params(n_images=n, diameter=diameter, band_height_deg=band, gap_deg=gap)
        assert make_buildable(p)
        panel = overlay.Panel.of(derive(p))
        for kind in ("wreath", "wreath_square", "candy"):
            slot = ImageSlot(path="photo.jpg", deco_frame=kind, deco_size=size)
            unit = overlay.frame_unit(slot, panel)
            assert unit <= overlay.deco_unit(slot, panel)
            dl, ph, hidden, light, _plain = coverage(slot, panel, 0.4)
            shown = float(((hidden < 0.01) * np.cos(ph)).sum() / np.cos(ph).sum())
            if kind == "wreath":            # a ring: its opening is a fair part of the picture's height
                ring_a, ring_b, bough = overlay.wreath_axes(panel, unit)
                tall = panel.band_half * panel.radius
                least = min(tall, panel.panel_half * panel.radius * math.cos(panel.band_half))
                assert min(ring_a, ring_b) - bough > 0.33 * least - 0.6, (n, diameter, band, gap, size)
                assert shown > 0.01         # (little of a very wide picture, but a real window)
            else:                           # along the edges: most of the picture is left
                assert shown > 0.15, (n, diameter, band, gap, size, kind, shown)
            # nothing of it lies beyond the picture: the bow is not cut off
            layer = overlay.layer_for(slot, panel, 0.3)
            dlam = np.linspace(-panel.panel_half, panel.panel_half, 300)
            for phi in (-panel.band_half - 0.4 / panel.radius, panel.band_half + 0.4 / panel.radius):
                assert layer.sample(dlam, np.full_like(dlam, phi))[1].max() < 0.2, (n, kind, size)
            if kind != "wreath":            # the border's line is a simple loop round the middle
                inset, corner = ((overlay.WREATH_BOUGH * unit + 0.3, 9.0 * unit) if kind == "wreath_square"
                                 else (overlay.CANDY_BAND * unit / 2 + overlay.EDGE_MM, 5.2 * unit))
                lon, lat, _along = overlay._inset_path(panel, inset, corner)
                u, v = imaging.panel_uv(lon, lat, panel.projection)
                turn = np.unwrap(np.arctan2(v, u))
                assert (np.diff(turn) > -1e-9).all(), (n, diameter, band, gap, size, kind)
    # on an ordinary picture the border is drawn at its own size
    assert overlay.frame_unit(ImageSlot(deco_frame="wreath"), PANEL) == overlay.deco_unit(ImageSlot(), PANEL)
    assert overlay.frame_unit(ImageSlot(deco_frame="line", deco_size=1.6), PANEL) == pytest.approx(1.6, abs=0.02)


def test_the_faint_warning_names_a_frame_thickness_that_clears_it():
    for step in range(8, 33):                                           # every stop of the Thinnest wall slider
        t_min = step * 0.05
        p = Params(n_images=3, ball_dividers="stars", t_min=t_min, t_max=max(2.8, t_min + 0.5), t_frame=t_min + 0.2)
        found = {c.title: c for c in checks.ball_checks(p, derive(p))}
        faint = found["Decorations on the ball will hardly show"]
        advised = float(faint.detail.split("Make the frame at least ")[1].split(" mm")[0])
        assert advised == pytest.approx(round(advised * 20) / 20)       # a stop of the Frame slider
        p.t_frame = advised
        assert "Decorations on the ball will hardly show" not in {c.title for c in checks.ball_checks(p, derive(p))}, t_min
    # a frame much THINNER than the thinnest wall shows them too (as darker shapes)
    thin = Params(n_images=3, ball_dividers="stars", t_min=1.6, t_max=2.8, t_frame=0.8)
    assert "Decorations on the ball will hardly show" not in {c.title for c in checks.ball_checks(thin, derive(thin))}


def test_with_the_relief_inside_the_ball_is_decorated_the_way_the_pictures_are(photo):
    """No ring round the bottom (that strip is the run-in to the closed end), the row where the
    inside joins the closing cone is never moved, and the checklist says how it will look."""
    p = with_photos(Params(n_images=3, relief_side="inside", ball_dividers="stars", ball_top="beads",
                           ball_bottom="zigzag", ball_pole="star"), photo)
    lay = derive(p)
    ball = overlay.ball_for(p, lay, 0.4)
    assert ball.no_room == ["ball_bottom"] and ball.bottom is None and ball.top is not None and ball.dividers
    assert overlay.ball_reach(p, lay)[0] == -lay.band_half              # no dense rows wasted down there
    found = {c.title: c for c in checks.ball_checks(p, lay)}
    assert "Choose Outside or Both" in found["No ring round the bottom with the picture on the inside"].detail
    said = found["Decorations on the ball"].detail
    assert "zigzag" not in said and "formed on the inside of the wall like the pictures" in said
    # with Outside the same ball has its ring
    p.relief_side = "outside"
    assert overlay.ball_for(p, derive(p), 0.4).no_room == []
    assert [c.level for c in checks.ball_checks(p, derive(p))] == [checks.OK]

    for extra in ({}, {"ball_dividers": "candy", "t_frame": 4.0, "t_max": 4.0}):
        plain = with_photos(Params(n_images=3, relief_side="inside", **{k: v for k, v in extra.items()
                                                                        if k != "ball_dividers"}), photo)
        decorated = with_photos(Params(n_images=3, relief_side="inside", ball_dividers="stars", ball_top="beads",
                                       ball_bottom="zigzag", **{k: v for k, v in extra.items()
                                                                if k != "ball_dividers"}), photo)
        if "ball_dividers" in extra:
            decorated.ball_dividers = extra["ball_dividers"]
        lay = derive(plain)
        photos, _ = imaging.photos_for(plain)
        a = geometry.build_body(plain, lay, photos, pitch=0.4, dense=True)
        b = geometry.build_body(decorated, lay, photos, pitch=0.4, dense=True)

        def junction(body):                 # the first ring of the inside above the closing cone
            inner = body.vertices[body.n_outer:]
            radius = np.linalg.norm(inner, axis=1)
            lat = np.arcsin(np.clip(inner[:, 2] / radius, -1, 1))
            return np.sort(radius[np.abs(lat + lay.phi_c) < 1e-6])

        assert len(junction(a)) > 100 and np.allclose(junction(a), junction(b), atol=1e-9), extra
