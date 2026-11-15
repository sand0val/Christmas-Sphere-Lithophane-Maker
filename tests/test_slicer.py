"""Tests for the Bambu Studio hand-off.

The real slicing test takes about half a minute and needs Bambu Studio, so it
only runs when asked for:

    set LSPHERE_SLICE_TEST=1
    python -m pytest tests/test_slicer.py -q
"""
from __future__ import annotations

import math
import os
import re
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithosphere import export, geometry, slicer  # noqa: E402
from lithosphere.params import (LIGHT_PRESETS, NOZZLES, Params, line_width,  # noqa: E402
                                make_buildable, recommended_layer_height)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import gcode_report  # noqa: E402


def test_names_and_durations():
    assert slicer.project_path_for(r"C:\x\my ornament.stl") == r"C:\x\my ornament_bambu_A1.3mf"
    assert slicer.project_path_for("a.b.3mf") == "a.b_bambu_A1.3mf"
    assert slicer.format_duration(17034) == "4 h 44 min"
    assert slicer.format_duration(1790) == "30 min"
    assert slicer.format_duration(None) == "unknown"


def test_expected_warnings_are_separated():
    floating = ("It seems object x.stl has floating regions. Please re-orient the object or enable "
                "support generation.")
    expected, unexpected = slicer.split_warnings([
        floating, "bed_temperature_too_high_than_filament", "something_new_and_alarming"])
    assert len(expected) == 1 and expected[0].startswith("bed_temperature")
    # a floating-regions warning is a real fault of the model, and comes with what to do
    assert len(unexpected) == 2 and unexpected[1] == "something_new_and_alarming"
    assert unexpected[0].startswith(floating) and "Print-safe edges" in unexpected[0]


def test_supports_note_only_mentions_a_warning_when_one_can_happen():
    assert "floating" not in export.supports_note(Params())
    assert "floating" not in export.supports_note(Params(relief_side="inside"))
    assert "floating" not in export.supports_note(Params(relief_side="split", safe_edges=False))
    for side in ("outside", "inside"):
        risky = Params(relief_side=side, safe_edges=False)
        assert "floating" in export.supports_note(risky) and export.relief_can_hang(risky)
    assert "Both" not in export.supports_note(Params(safe_edges=False))    # no longer the remedy
    sheet = export.settings_sheet(Params(), geometry.derive(Params()))
    assert "Expected warning" not in sheet and "Keep supports OFF" in sheet


def test_process_settings_are_consistent():
    p = Params()
    overrides = export.process_overrides(p)
    assert overrides["enable_support"] is False
    assert overrides["sparse_infill_density"] == "100%"
    # Bambu Studio refuses gyroid (the High Quality preset's pattern) at 100 % density
    assert overrides["sparse_infill_pattern"] == "zig-zag"
    assert overrides["wall_loops"] == export.wall_loops_for(p) >= 5
    assert export.process_preset(p) == "0.12mm High Quality @BBL A1"
    assert export.process_preset(Params(flare_angle=25)) == "0.08mm High Quality @BBL A1"
    shown = {name: value for name, value, _why in export.slicer_settings(p)}
    assert shown["Wall loops"] == str(overrides["wall_loops"])
    assert shown["Process preset"] == export.process_preset(p)
    assert 3.0 < export.estimated_print_hours(p, 40.0) < 6.0
    # the same overrides go on top of the fine nozzle's own preset
    fine = Params(nozzle="0.2")
    assert export.process_overrides(fine) == {**overrides, "wall_loops": export.wall_loops_for(fine)}
    shown = {name: value for name, value, _why in export.slicer_settings(fine)}
    assert shown["Printer / nozzle"] == "Bambu Lab A1, 0.2 mm nozzle"
    assert shown["Process preset"] == export.process_preset(fine) and shown["Layer height"] == "0.06 mm"


@pytest.mark.skipif(slicer.find_bambu_studio() is None, reason="Bambu Studio is not installed")
def test_bambu_studio_has_the_presets_each_nozzle_names():
    """The presets the project is built from must exist in the installed Bambu Studio,
    belong together and really have the layer height and line width the design assumes."""
    from lithosphere import bambu_cli

    exe = slicer.find_bambu_studio()
    for key, nozzle in NOZZLES.items():
        machine = bambu_cli.flatten_preset("machine", nozzle.machine, exe=exe)
        assert float(machine["nozzle_diameter"][0]) == nozzle.diameter, key
        filament = bambu_cli.flatten_preset("filament", nozzle.filament, exe=exe)
        assert nozzle.machine in filament["compatible_printers"], key
        assert len(nozzle.processes) == len(nozzle.layers)
        for layer, name in zip(nozzle.layers, nozzle.processes):
            process = bambu_cli.flatten_preset("process", name, exe=exe)
            assert nozzle.machine in process["compatible_printers"], name
            assert float(process["layer_height"]) == layer, name
            assert float(process["outer_wall_line_width"]) == nozzle.line_width, name
            assert float(process["outer_wall_speed"][0]) <= 60.0, name      # the slow, clean variant
            lo, hi = float(machine["min_layer_height"][0]), float(machine["max_layer_height"][0])
            assert lo <= layer <= hi, name


def test_gcode_report_reads_features_and_seams():
    # two layers of a square loop that starts at the back (+Y) of a part centred at (128, 128)
    lines = ["; total layer number: 2", "M83"]
    for z in (0.2, 0.32):
        lines += ["; CHANGE_LAYER", f"; Z_HEIGHT: {z}", "G1 X128 Y148", "; FEATURE: Outer wall",
                  "G1 X148 Y148 E1", "G1 X148 Y108 E2", "G1 X108 Y108 E2", "G1 X108 Y148 E2",
                  "G1 X128 Y148 E1", "; FEATURE: Bridge", "G1 X120 Y120", "G1 X130 Y120 E0.5"]
    data = gcode_report.analyse("\n".join(lines))
    assert len(data["layers"]) == 2
    assert data["per_feature"]["Outer wall"] == pytest.approx(16.0)
    assert data["per_feature"]["Bridge"] == pytest.approx(1.0)
    seams = gcode_report.seam_stats(data)
    assert seams["loops"] == 2 and seams["within"][3] == 100.0


def _ring(z: float, start_deg: float, radius: float = 38.0, feature: str = "Outer wall",
          close: bool = True, sides: int = 72) -> list[str]:
    """G-code of one wall loop round a part centred on the bed, starting start_deg
    away from the back (+Y), the way Bambu Studio writes it."""
    def at(k: int) -> tuple[float, float]:
        a = math.radians(start_deg + 360.0 * k / sides)
        return 128.0 + radius * math.sin(a), 128.0 + radius * math.cos(a)

    x, y = at(0)
    lines = [f"; Z_HEIGHT: {z}", f"G1 X{x:.3f} Y{y:.3f} F42000", f"G1 Z{z}", "G1 E.8 F1800",
             f"; FEATURE: {feature}", "G1 F3600"]
    for k in range(1, sides + 1 if close else sides // 2):
        x, y = at(k)
        lines.append(f"G1 X{x:.3f} Y{y:.3f} E.05")
    return lines + ["; WIPE_START", f"G1 X{x - 1:.3f} Y{y:.3f} E-.2", "; WIPE_END"]


def _fake_project(path, rings: list[list[str]]) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Metadata/plate_1.gcode", "\n".join(["M83"] + [ln for r in rings for ln in r]))
    return str(path)


def test_wall_loop_starts_finds_the_seams():
    gcode = "\n".join(["M83"]
                      + _ring(40.0, 0.0)                              # a loop that starts at the back
                      + _ring(40.12, 9.0)                             # one that starts 9 deg off
                      + _ring(40.24, 30.0, feature="Inner wall")      # not an outside surface
                      + _ring(40.36, 50.0, close=False)               # an open path is no loop
                      + _ring(40.48, 70.0, radius=3.0))               # too short: a picture detail
    starts = slicer.wall_loop_starts(gcode)
    assert [z for z, _x, _y in starts] == [40.0, 40.12]
    (_z, x0, y0), (_z, x1, y1) = starts
    assert (x0, y0) == pytest.approx((128.0, 166.0), abs=1e-3)
    assert math.degrees(math.atan2(x1 - 128.0, y1 - 128.0)) == pytest.approx(9.0, abs=1e-3)

    # the rim of a dip in a picture is a closed outer wall too, but it does not go round the
    # ornament, so it is not one of its seams
    dip = [f"; Z_HEIGHT: 41.0", "G1 X150 Y150 F42000", "; FEATURE: Outer wall"]
    dip += [f"G1 X{150 + 8 * math.sin(a):.3f} Y{150 + 8 * math.cos(a) - 8:.3f} E.05"
            for a in np.linspace(0, 2 * math.pi, 49)[1:]]
    assert slicer.wall_loop_starts("\n".join(["M83"] + dip + ["G1 X0 Y0"])) == []

    # Bambu Studio writes smooth walls as arcs: two half circles are still one loop round
    arcs = ["M83", "; Z_HEIGHT: 42.0", "G1 X128 Y166 F42000", "; FEATURE: Outer wall",
            "G2 X128 Y90 I0 J-38 E2", "G2 X128 Y166 I0 J38 E2", "G1 X0 Y0"]
    assert slicer.wall_loop_starts("\n".join(arcs)) == [(42.0, 128.0, 166.0)]
    assert slicer.wall_loop_starts("\n".join(arcs).replace("G2", "G3")) == [(42.0, 128.0, 166.0)]

    # ... and now and then puts a move of no length and no plastic in the middle of a loop
    # (seen with the 0.2 mm nozzle: "G1 X143.452 Y93.38 E0").  That is not a travel move.
    ring = _ring(43.0, 2.0)
    moves = [k for k, line in enumerate(ring) if " E.05" in line]
    middle = moves[len(moves) // 2]
    stutter = ring[:middle + 1] + [ring[middle].replace(" E.05", " E0")] + ring[middle + 1:]
    assert len(slicer.wall_loop_starts("\n".join(["M83"] + ring))) == 1
    assert len(slicer.wall_loop_starts("\n".join(["M83"] + stutter))) == 1
    # ... or one a thousandth of a millimetre long, too short to be given any plastic
    x, y = (float(v) for v in re.match(r"G1 X([\d.]+) Y([\d.]+)", ring[middle]).groups())
    creep = ring[:middle + 1] + ["G1 F2700", f"G1 X{x:.3f} Y{y + 0.001:.3f} E0"] + ring[middle + 1:]
    assert len(slicer.wall_loop_starts("\n".join(["M83"] + creep))) == 1
    # Now and then Bambu Studio prints a loop in two passes: it stops part of the way round,
    # prints a small separate ridge of the relief somewhere else, comes back to where it
    # stopped and finishes the loop.  That is still one loop right round - the wall is whole -
    # with a second place where the printer starts.
    back = ring[middle].replace(" E.05", "")
    hop = ring[:middle + 1] + ["G1 X0 Y0 F42000", back] + ring[middle + 1:]
    (z, starts), = slicer.wall_loops("\n".join(["M83"] + hop))
    assert z == 43.0 and len(starts) == 2
    assert starts[0] == slicer.wall_loop_starts("\n".join(["M83"] + ring))[0][1:]
    assert starts[1] == pytest.approx((x, y), abs=1e-3)
    assert slicer.wall_loop_starts("\n".join(["M83"] + hop)) == [(43.0,) + starts[0]]
    # the same with the ridge printed in between, and with a retraction that moves (a wipe)
    island = ["; FEATURE: Outer wall", "G1 X30 Y30 F42000", "G1 X31 Y30 E.05", "G1 X31 Y31 E.05",
              "G1 X30 Y30 E.05"]
    detour = ring[:middle + 1] + island + [back] + ring[middle + 1:]
    assert [len(starts) for _z, starts in slicer.wall_loops("\n".join(["M83"] + detour))] == [2]
    wipe = ring[:middle + 1] + [f"G1 X{x:.3f} Y{y + 0.001:.3f} E-.2"] + ring[middle + 1:]
    assert [len(starts) for _z, starts in slicer.wall_loops("\n".join(["M83"] + wipe))] == [2]
    # a ridge that happens to start where the first pass stopped does not hide the loop, and
    # the seam is still where the FIRST pass began
    twig = ["; FEATURE: Outer wall", f"G1 X{x + 0.3:.3f} Y{y:.3f} F42000", f"G1 X{x + 2:.3f} Y{y + 2:.3f} E.05"]
    beside = ring[:middle + 1] + twig + [back] + ring[middle + 1:]
    (z, starts), = slicer.wall_loops("\n".join(["M83"] + beside))
    assert len(starts) == 2 and starts[0] == slicer.wall_loop_starts("\n".join(["M83"] + ring))[0][1:]
    # a loop with a piece missing is no loop: the printer comes back further on
    gap = ring[:middle + 1] + ["G1 X0 Y0 F42000", ring[middle + 4].replace(" E.05", "")] + ring[middle + 5:]
    assert slicer.wall_loops("\n".join(["M83"] + gap)) == []
    # an unbroken loop has one start
    assert [len(starts) for _z, starts in slicer.wall_loops("\n".join(["M83"] + ring))] == [1]


def test_seam_summary_says_whether_the_seam_hides_on_the_divider(tmp_path):
    lay = geometry.derive(Params())                     # 12 deg gap: the divider is +-6 deg
    z0 = lay.z_top
    hidden = _fake_project(tmp_path / "a.3mf", [_ring(z0 - 20 + k, 0.5 * (-1) ** k) for k in range(40)])
    seam = slicer.seam_summary(hidden, lay)
    assert seam["loops"] == 40 and seam["stray"] == 0 and seam["on_divider"] == 1.0
    assert seam["divider_deg"] == pytest.approx(12.0) and seam["worst_deg"] == pytest.approx(0.5, abs=1e-3)
    assert seam["worst_mm"] == pytest.approx(math.radians(0.5) * 38.0, abs=1e-3)
    ok, text = slicer.seam_note(seam)
    assert ok and "down the middle of the divider" in text and "0.3 mm" in text

    # a quarter of the layers start 9 deg off, in the picture; layers outside the pictures do not count
    rings = [_ring(z0 - 20 + k, 9.0 if k % 4 == 0 else 0.0) for k in range(40)]
    rings += [_ring(z0 + 37.0, 90.0), _ring(z0 - 37.0, 90.0)]
    stray = slicer.seam_summary(_fake_project(tmp_path / "b.3mf", rings), lay)
    assert stray["loops"] == 40 and stray["stray"] == 10
    assert stray["on_divider"] == pytest.approx(0.75) and stray["worst_deg"] == pytest.approx(9.0, abs=1e-3)
    ok, text = slicer.seam_note(stray)
    assert not ok and "10 of the 40 wall loops" in text and "wider gap" in text
    # even a single stray loop is reported
    one = dict(seam, stray=1, on_divider=39 / 40)
    assert not slicer.seam_note(one)[0] and "1 of the 40" in slicer.seam_note(one)[1]

    # a loop that starts nowhere near the divider is not the seam wandering off it: now and
    # then Bambu Studio starts a wall loop at a small island of the relief.  That is said
    # plainly, without the advice about the gap - and is only a fault when it is more than a few
    rings = [_ring(z0 - 20 + 0.1 * k, 95.0 if k in (50, 51) else 0.3) for k in range(400)]
    odd = slicer.seam_summary(_fake_project(tmp_path / "d.3mf", rings), lay)
    assert odd["loops"] == 400 and odd["stray"] == 2 and odd["elsewhere"] == 2
    ok, text = slicer.seam_note(odd)
    assert ok and "except for 2 of the 400" in text and "ridge of the relief" in text
    assert "wider gap" not in text and "Smoothing" in text
    ok, text = slicer.seam_note(dict(odd, stray=1, elsewhere=1))
    assert ok and "except for 1 of the 400" in text and "that one" in text and "It leaves" in text
    ok, text = slicer.seam_note(dict(odd, stray=40, elsewhere=40))      # a tenth of them: a fault
    assert not ok and "except for 40 of the 400" in text
    ok, text = slicer.seam_note(dict(odd, stray=12, elsewhere=2))       # most are beside the divider
    assert not ok and "12 of the 400 wall loops" in text and "wider gap" in text
    assert stray["elsewhere"] == 0 and seam["elsewhere"] == 0

    # no gap at all: the seam has to be between two pictures, and that is not a fault
    flush = geometry.derive(Params(gap_deg=0.0))
    seam = slicer.seam_summary(hidden, flush)
    ok, text = slicer.seam_note(seam)
    assert ok and "no gap" in text
    assert slicer.seam_note(None) == (True, "") and slicer.seam_note({"loops": 0}) == (True, "")


def test_seam_summary_counts_layers_that_are_not_a_closed_ring(tmp_path):
    """Each layer of the pictures should be two wall loops right round: the outside and the
    inside of the wall.  A layer where the slicer joined them (the wall was too thin for two
    lines somewhere) has none, and is counted."""
    lay = geometry.derive(Params())
    z0 = lay.z_top

    def layer(k: int, sound: bool = True) -> list[str]:
        z = round(z0 - 10 + 0.12 * k, 2)
        if sound:
            return _ring(z, 0.2) + _ring(z, -0.2, radius=36.0)[1:]      # one Z_HEIGHT line a layer
        # one closed outer wall that runs out along the outside, back along the inside
        # and never goes round the axis: the two loops joined at a thin spot
        points = [(128.0 + r * math.sin(math.radians(a)), 128.0 + r * math.cos(math.radians(a)))
                  for r, sweep in ((38.0, range(2, 358, 4)), (36.0, range(358, 2, -4))) for a in sweep]
        lines = [f"; Z_HEIGHT: {z}", f"G1 X{points[0][0]:.3f} Y{points[0][1]:.3f} F42000",
                 "; FEATURE: Outer wall"]
        lines += [f"G1 X{x:.3f} Y{y:.3f} E.05" for x, y in points[1:] + points[:1]]
        return lines + ["G1 X0 Y0 F42000"]

    sound = slicer.seam_summary(_fake_project(tmp_path / "a.3mf", [layer(k) for k in range(60)]), lay)
    assert sound["layers"] == 60 and sound["broken"] == 0 and sound["loops"] == 120
    assert slicer.ring_note(sound) == (True, "")
    rings = [layer(k, sound=k not in (7, 8, 30)) for k in range(60)]
    flawed = slicer.seam_summary(_fake_project(tmp_path / "b.3mf", rings), lay)
    assert flawed["layers"] == 60 and flawed["broken"] == 3 and flawed["loops"] == 114
    assert flawed["stray"] == 0                         # where the seam is, is another matter
    ok, text = slicer.ring_note(flawed)
    assert not ok and "3 of the 60 layers" in text and "thinnest wall" in text
    # layers above and below the pictures (the cap, the closed end) are not counted
    far = [layer(k) for k in range(60)] + [_ring(z0 + 37.0, 0.0), _ring(z0 - 37.0, 0.0)]
    assert slicer.seam_summary(_fake_project(tmp_path / "c.3mf", far), lay)["broken"] == 0
    assert slicer.ring_note(None) == (True, "") and slicer.ring_note({"loops": 0}) == (True, "")


def test_a_loop_printed_in_two_passes_is_whole_but_leaves_a_second_start(tmp_path):
    """Seen with lettering and decorations: in one layer of 511 the outside loop was printed in
    two passes.  The wall is whole, so the layer is not counted as broken; the place where the
    second pass starts is counted only if it lies in a picture."""
    lay = geometry.derive(Params())                     # three pictures: dividers at 0 and +-120 deg
    z0 = lay.z_top

    def layer(k: int, stop_deg: float | None = None) -> list[str]:
        z = round(z0 - 10 + 0.12 * k, 2)
        outside = _ring(z, 0.2)
        if stop_deg is not None:                        # stop there, go away, come back, carry on
            moves = [i for i, line in enumerate(outside) if " E.05" in line]
            cut = moves[int(round(stop_deg / 5.0)) - 1]
            outside = (outside[:cut + 1] + ["G1 X0 Y0 F42000", outside[cut].replace(" E.05", "")]
                       + outside[cut + 1:])
        return outside + _ring(z, -0.2, radius=36.0)[1:]

    whole = slicer.seam_summary(_fake_project(tmp_path / "a.3mf", [layer(k) for k in range(50)]), lay)
    assert whole["broken"] == 0 and whole["resumed"] == 0 and whole["loops"] == 100
    # second start 90 degrees round: inside a picture
    rings = [layer(k, 90.0 if k == 20 else None) for k in range(50)]
    seam = slicer.seam_summary(_fake_project(tmp_path / "b.3mf", rings), lay)
    assert seam["broken"] == 0 and seam["loops"] == 100 and seam["stray"] == 0 and seam["resumed"] == 1
    assert slicer.ring_note(seam) == (True, "")
    ok, text = slicer.seam_note(seam)
    assert ok and "except for 1 of the 100" in text and "two passes" in text
    # second start 120 degrees round: on a divider, where it does not show
    rings = [layer(k, 120.0 if k == 20 else None) for k in range(50)]
    seam = slicer.seam_summary(_fake_project(tmp_path / "c.3mf", rings), lay)
    assert seam["broken"] == 0 and seam["resumed"] == 0
    assert "never more than" in slicer.seam_note(seam)[1]
    # many of them are a fault
    ok, text = slicer.seam_note(dict(seam, resumed=30))
    assert not ok and "except for 30 of the 100" in text


def test_settings_sheet_explains_the_seam():
    sheet = export.settings_sheet(Params(), geometry.derive(Params()))
    assert "\nSeam\n" in sheet and "Seam painting" in sheet and "An STL cannot carry" in sheet


@pytest.mark.skipif(not (slicer.can_slice() and os.environ.get("LSPHERE_SLICE_TEST") == "1"),
                    reason="set LSPHERE_SLICE_TEST=1 (and install Bambu Studio) to run the real slice")
@pytest.mark.parametrize("side,nozzle,light", [
    ("split", "0.4", "mini"), ("inside", "0.4", "mini"), ("outside", "0.4", "mini"),
    ("outside", "0.2", "mini"),             # the fine nozzle
    ("outside", "0.4", "puck"),             # a pocket for a puck light in the cap
    ("outside", "0.2", "puck"),
])
def test_real_slice_is_clean(tmp_path, capsys, side, nozzle, light):
    """Bambu Studio itself must find nothing to warn about - in particular no
    "floating regions" - with a photo made of the hardest edges there are."""
    rng = np.random.default_rng(11)
    blocks = np.kron(rng.choice([0, 255], size=(30, 30)), np.ones((12, 12)))
    photo = str(tmp_path / "harsh.png")
    Image.fromarray(blocks.astype(np.uint8)).save(photo)
    p = Params(n_images=3, quality="draft", relief_side=side, nozzle=nozzle)
    preset = LIGHT_PRESETS[light]
    if preset.seat is not None:
        p.light_preset, p.hole_diameter = light, preset.hole
        p.seat, p.seat_diameter, p.seat_depth = True, preset.seat, preset.depth
        p.cap_diameter = preset.seat + 5.0
        assert make_buildable(p)
    for slot in p.images:
        slot.path = photo
    layer = recommended_layer_height(p)
    build = geometry.build_ornament(p)
    out = str(tmp_path / "ornament_bambu_A1.3mf")
    result = slicer.slice_mesh(build.vertices, build.faces, out, p, seam_faces=build.seam_faces)
    assert result["ok"], result.get("error")
    assert result["support_used"] is False
    assert result["layer_height"] == layer
    assert result["layer_count"] == pytest.approx(build.layout.height / layer, abs=4)
    assert result["presets"]["machine"] == NOZZLES[nozzle].machine
    assert result["presets"]["filament"] == [NOZZLES[nozzle].filament]
    assert result["presets"]["process"].startswith(export.process_preset(p))
    assert not result["overrides_mismatched"]
    assert abs(result["filament_g"] - build.stats["weight_g"]) / build.stats["weight_g"] < 0.08
    assert not any("floating" in text for text in result["warnings"]), result["warnings"]
    _expected, unexpected = slicer.split_warnings(result["warnings"])
    assert not unexpected, unexpected

    # the project keeps the exported orientation (identity transform: no auto-arrange turn)
    # and the seam line, so slicing it again in Bambu Studio gives the same seam
    with zipfile.ZipFile(out) as zf:
        model = zf.read("3D/3dmodel.model").decode()
        painted = sum(zf.read(name).decode("utf-8", "replace").count('paint_seam="4"')
                      for name in zf.namelist() if name.startswith("3D/Objects/"))
    transforms = re.findall(r'<(?:item|component) [^>]*transform="([^"]+)"', model)
    assert len(transforms) == 2, model
    for transform in transforms:                    # 3 x 3 rotation, then the position
        assert transform.split()[:9] == "1 0 0 0 1 0 0 0 1".split(), transform
    assert painted == int(build.seam_faces.sum()) > 0

    # every layer of the pictures starts on the divider at the back, inside and outside:
    # two loops a layer, and (nearly) all of them are found and checked
    seam = result["seam"]
    band_layers = 2 * build.layout.R * math.sin(build.layout.band_half) / layer
    assert seam["layers"] == pytest.approx(band_layers, abs=2), seam
    # No seam wanders off the divider into the edge of a picture.  What this photo can
    # provoke is something else: its blocks end in ridges that a layer cuts off as small
    # islands, and now and then Bambu Studio starts a wall loop at one of those instead
    # (seen: 1 to 3 loops in 1,000 to 2,000).  That is the slicer's doing, it is reported
    # for what it is, and it has to stay that rare.
    assert seam["stray"] == seam["elsewhere"] <= 0.003 * seam["loops"], seam
    if not seam["stray"]:
        assert seam["on_divider"] == 1.0 and seam["worst_mm"] < 1.5, seam
    hidden, text = slicer.seam_note(seam)
    assert hidden and "divider at the back" in text
    # ... and every layer is a closed ring of two loops: nowhere is the wall so thin that the
    # slicer joins the outside and the inside loop (the hardest edges there are, remember)
    assert seam["broken"] == 0 and seam["loops"] == 2 * seam["layers"], seam
    assert slicer.ring_note(seam) == (True, "")

    with capsys.disabled():
        data = gcode_report.report(out)
    assert not data["supports"]
    assert data["wall_percent"] > 95.0
    bridges = data["risky"].get("Bridge", [])
    assert len(bridges) == 1, "exactly one bridged layer: the ceiling of the closed end"
    ceiling = build.layout.z_top + build.layout.z_f
    assert abs(bridges[0][0] - ceiling) < 0.3
    # over the whole print as well, every loop starts at the back - but for the same few
    assert data["seams"]["within"][6] >= 99.7
    if not seam["stray"]:
        assert data["seams"]["within"][6] == 100.0
    assert line_width(p) == NOZZLES[nozzle].line_width


@pytest.mark.skipif(not (slicer.can_slice() and os.environ.get("LSPHERE_SLICE_TEST") == "1"),
                    reason="set LSPHERE_SLICE_TEST=1 (and install Bambu Studio) to run the real slice")
@pytest.mark.parametrize("nozzle", ["0.4", "0.2"])
def test_real_slice_with_lettering_and_decorations_is_clean(tmp_path, capsys, nozzle):
    """Lettering and decorations are nothing but sudden edges, top to bottom of a picture -
    and with the plain parts of the ball decorated too, from the shoulder to the closed end.
    Bambu Studio must still slice the ornament without supports and without a warning, with
    every layer that carries relief a whole double wall and the seam on the divider - but for
    the odd loop it starts at a ridge of the relief or prints in two passes, which has to stay
    rare."""
    photo = str(tmp_path / "ramp.png")
    Image.fromarray(np.tile(np.linspace(20, 235, 300), (240, 1)).astype(np.uint8)).save(photo)
    p = Params(n_images=4, quality="draft", nozzle=nozzle)
    a, b, c, d = p.images
    a.path = b.path = c.path = photo
    a.text, a.text_y, a.deco_top, a.deco_bottom = "Happy Holidays", -1.0, "garland", "garland"
    b.text, b.text_y, b.deco_top, b.deco_scatter = "Johnny & Rachel\n2026", 1.0, "lights", "snowflakes"
    c.deco_frame, c.deco_scatter, c.deco_bottom = "wreath", "stars", "holly"
    d.text, d.text_y, d.backdrop = "Happy Holidays\nfrom\nJohnny and Rachel\n2026", 0.0, "dark"
    d.deco_top, d.deco_bottom, d.deco_scatter, d.deco_frame = "icicles", "trees", "hearts", "candy"
    a.deco_frame = "wreath_square"
    p.ball_dividers, p.ball_top, p.ball_bottom, p.ball_pole = "stars", "zigzag", "snowflakes", "snowflake"
    layer = recommended_layer_height(p)
    build = geometry.build_ornament(p)
    assert not build.problems
    out = str(tmp_path / "ornament_bambu_A1.3mf")
    result = slicer.slice_mesh(build.vertices, build.faces, out, p, seam_faces=build.seam_faces)
    assert result["ok"], result.get("error")
    assert result["support_used"] is False and result["layer_height"] == layer
    assert not any("floating" in text for text in result["warnings"]), result["warnings"]
    _expected, unexpected = slicer.split_warnings(result["warnings"])
    assert not unexpected, unexpected
    assert abs(result["filament_g"] - build.stats["weight_g"]) / build.stats["weight_g"] < 0.08

    seam = result["seam"]
    assert seam["broken"] == 0 and seam["loops"] == 2 * seam["layers"], seam
    # (the rings above and below the pictures are read as well: more layers than the pictures have)
    from lithosphere.params import derive
    lay = derive(p)
    assert seam["layers"] > 1.1 * 2 * lay.R * math.sin(lay.band_half) / layer, seam
    assert slicer.ring_note(seam) == (True, "")
    assert seam["stray"] == seam["elsewhere"], seam            # none beside the divider
    assert seam["elsewhere"] + seam["resumed"] <= 0.005 * seam["loops"], seam
    hidden, text = slicer.seam_note(seam)
    assert hidden and "divider at the back" in text

    with capsys.disabled():
        data = gcode_report.report(out)
    assert not data["supports"] and data["wall_percent"] > 95.0
    assert len(data["risky"].get("Bridge", [])) == 1           # the ceiling of the closed end
    assert "Overhang wall" not in data["risky"]
