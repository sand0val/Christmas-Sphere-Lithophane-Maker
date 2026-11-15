"""Mesh file writers (binary STL, 3MF) and the print-settings sheet."""
from __future__ import annotations

import os
import zipfile
from xml.sax.saxutils import escape

import numpy as np

from . import APP_NAME, __version__
from .params import (DECO_BOTTOM, DECO_FRAME, DECO_SCATTER, DECO_TOP, PRINTER_NAME, Layout, Params,
                     band_half_limit, line_width, nozzle_for, recommended_layer_height)

EXPORT_FILTER = "STL mesh (*.stl);;3MF model (*.3mf)"


def write_stl(path: str, vertices: np.ndarray, faces: np.ndarray, name: str = "ornament") -> None:
    """Binary STL (millimetres)."""
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(length > 0, length, 1.0)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    header = f"{APP_NAME} {__version__} - {name}".encode("ascii", "replace")[:80].ljust(80, b" ")
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(header)
        fh.write(np.uint32(len(faces)).tobytes())
        rec.tofile(fh)
    os.replace(tmp, path)


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
    ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
    '</Types>\n')
_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
    '</Relationships>\n')


def write_3mf(path: str, vertices: np.ndarray, faces: np.ndarray, name: str = "ornament",
              seam_faces: np.ndarray | None = None) -> None:
    """Plain 3MF (core spec): much smaller than STL and carries its units.

    seam_faces (one True / False per triangle) marks triangles as a seam
    enforcer, written the way Bambu Studio's own "Seam painting" tool stores it
    (paint_seam="4").  Bambu Studio and OrcaSlicer then start every layer on
    those triangles; programs that do not know the attribute ignore it."""
    v_lines = ['<vertex x="%.4f" y="%.4f" z="%.4f"/>' % tuple(row) for row in vertices.tolist()]
    f_lines = ['<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(row) for row in faces.tolist()]
    if seam_faces is not None:
        for k in np.flatnonzero(np.asarray(seam_faces, dtype=bool)).tolist():
            f_lines[k] = f_lines[k][:-2] + ' paint_seam="4"/>'
    model = "".join([
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n',
        f' <metadata name="Title">{escape(name)}</metadata>\n',
        f' <metadata name="Application">{escape(APP_NAME)} {__version__}</metadata>\n',
        ' <resources>\n',
        f'  <object id="1" type="model" name="{escape(name, {chr(34): "&quot;"})}">\n',
        '   <mesh>\n    <vertices>\n',
        "\n".join(v_lines),
        '\n    </vertices>\n    <triangles>\n',
        "\n".join(f_lines),
        '\n    </triangles>\n   </mesh>\n  </object>\n </resources>\n',
        ' <build>\n  <item objectid="1"/>\n </build>\n</model>\n',
    ])
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("3D/3dmodel.model", model)
    os.replace(tmp, path)


def write_mesh(path: str, vertices: np.ndarray, faces: np.ndarray,
               seam_faces: np.ndarray | None = None) -> None:
    """Write STL or 3MF depending on the file extension.  Only a 3MF can carry
    the seam line (see write_3mf); an STL is triangles and nothing else."""
    name = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".3mf"):
        write_3mf(path, vertices, faces, name, seam_faces)
    else:
        write_stl(path, vertices, faces, name)


# ---------------------------------------------------------------------------
# print settings
# ---------------------------------------------------------------------------
def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width)


def wall_loops_for(p: Params) -> int:
    """Enough wall loops that even the thickest wall is printed as concentric
    walls only (no infill pattern showing through the picture).

    A layer cuts the leaning wall at an angle, so at the top and bottom of the
    pictures the wall is 1 / cos(latitude) wider in the layer than it is thick;
    the loops (which grow in from both faces) have to cover that width."""
    band_half = min(np.radians(p.band_height_deg) / 2, band_half_limit(p))
    widest = max(max(p.t_max, p.t_frame) / np.cos(band_half), p.cap_wall)
    return int(np.ceil(widest / (2 * line_width(p)) + 0.25))


def process_preset(p: Params) -> str:
    """Name of the Bambu Studio process preset to start from."""
    nozzle = nozzle_for(p)
    return nozzle.processes[nozzle.layers.index(recommended_layer_height(p))]


def machine_preset(p: Params) -> str:
    """Name of the Bambu Studio printer preset (it names the nozzle)."""
    return nozzle_for(p).machine


def filament_preset(p: Params) -> str:
    """Name of the Bambu Studio filament preset that goes with the nozzle."""
    return nozzle_for(p).filament


def process_overrides(p: Params) -> dict[str, object]:
    """Settings changed on top of process_preset(), by their Bambu Studio keys.
    This is exactly what the built-in "Slice now" uses."""
    return {
        "wall_loops": wall_loops_for(p),
        "sparse_infill_density": "100%",
        "sparse_infill_pattern": "zig-zag",        # "Rectilinear"; gyroid is refused at 100 %
        "wall_generator": "arachne",
        "seam_position": "back",
        "enable_support": False,
        "brim_type": "outer_only",
        "brim_width": 5,
    }


# Lettering and decorations are sudden edges all the way round their outlines,
# and every one of them makes the printer slow down and turn: a decorated
# picture costs about as much time as this many cm3 of plain plastic.
DECORATED_PICTURE_CM3 = 0.94
# A wreath is some two hundred needles, a candy-cane border fifty stripes: a
# picture framed by one costs this much more again.
FRAME_CM3 = {"wreath": 1.2, "wreath_square": 1.2, "candy": 0.7}
# The same for what is drawn on the plain parts of the ball: a ring round the
# top or the bottom, each decorated divider, the shape underneath.
DECORATED_RING_CM3 = 0.9
DECORATED_DIVIDER_CM3 = 0.5
DECORATED_POLE_CM3 = 0.3


def _ball_cm3(p: Params) -> float:
    """What the decorations on the plain parts of the ball cost in print time,
    as so many cm3 of plain plastic (0 if there are none, or none fit)."""
    if not p.ball_decorated:
        return 0.0
    from . import overlay
    from .params import GeometryError, derive

    try:
        lay = derive(p)
    except GeometryError:
        return 0.0
    ball = overlay.ball_for(p, lay, 1.0)        # (drawn coarsely: only what there is matters)
    if ball is None:
        return 0.0
    return (DECORATED_DIVIDER_CM3 * len(ball.dividers)
            + DECORATED_RING_CM3 * sum(1 for ring in (ball.top, ball.bottom) if ring is not None)
            + (DECORATED_POLE_CM3 if ball.pole is not None else 0.0))


def estimated_print_hours(p: Params, volume_cm3: float) -> float:
    """Rough print time.  Calibrated on Bambu Studio slices of this design
    (High Quality presets, walls only): just under 8 minutes per cm3 of plastic
    with the 0.4 mm nozzle at 0.12 mm layers, 23 minutes with the 0.2 mm nozzle
    at 0.06 mm layers, plus 7 and 21 minutes for each picture that carries
    lettering or decorations (more for a wreath or a candy-cane border) - and
    about as much again for each ring of decoration round the ball, half of it
    for each decorated divider.  Seven slices - drawn pictures and photos,
    plain and decorated, both nozzles - came out within 7 % of it, and so did
    eight with the new borders and the ball decorated."""
    nozzle = nozzle_for(p)
    pictures = sum(DECORATED_PICTURE_CM3 + FRAME_CM3.get(slot.deco_frame, 0.0)
                   for slot in p.images[:p.n_images] if slot.decorated)
    return ((volume_cm3 + pictures + _ball_cm3(p)) * nozzle.hours_per_cm3
            * (nozzle.layers[0] / recommended_layer_height(p)))


def slicer_settings(p: Params) -> list[tuple[str, str, str]]:
    """(setting, value, why) rows for Bambu Studio."""
    layer = recommended_layer_height(p)
    nozzle = nozzle_for(p)
    return [
        ("Printer / nozzle", f"{PRINTER_NAME}, {nozzle.diameter:g} mm nozzle",
         "" if p.nozzle == "0.4" else "fit this nozzle and select it on the printer first"),
        ("Filament", "White PLA (e.g. Bambu PLA Basic white)", "matte or silk PLA is too opaque"),
        ("Process preset", process_preset(p), "outer walls at 60 mm/s; then change the settings below"),
        ("Layer height", f"{layer:.2f} mm", "already set by the preset"),
        ("Wall loops", str(wall_loops_for(p)), "walls only - no infill pattern in the picture"),
        ("Sparse infill density", "100 %", "anything the walls do not fill must be solid"),
        ("Sparse infill pattern", "Rectilinear", "Bambu Studio refuses gyroid at 100 %"),
        ("Wall generator", "Arachne", "varies line width to fill the changing wall thickness"),
        ("Seam position", "Back", "the project also marks a seam line down the back divider"),
        ("Supports", "Off", "nothing starts in mid-air printed cap-down"),
        ("Brim", "Outer brim only, 5 mm", "keeps the tall part stuck to the moving bed"),
        ("Orientation", "As exported - cap on the bed", "do not rotate or auto-orient it"),
    ]


SUPPORTS_NOTE = (
    "Keep supports OFF. Printed cap-down, no part of this ornament starts in mid-air, so "
    "Bambu Studio has nothing to warn about - and supports inside the ball could never be "
    "removed.")
UNSAFE_EDGES_NOTE = (
    "Print-safe edges is switched off, so Bambu Studio may warn that the object \"has floating "
    "regions\". Take that warning seriously: those are picture details that start in mid-air "
    "and print as blobs. Switch Print-safe edges back on (Light & dark page).")


SEAM_NOTE = (
    "The ready-to-print Bambu Studio project and the .3mf model mark a line down the middle of "
    "the divider at the back as the seam, and every layer starts on it. An STL cannot carry "
    "that marking. If you slice the plain STL yourself, remember that Bambu Studio's Back seam "
    "is only roughly at the back: it can settle well to one side of the divider, in the edge of "
    "a picture. Draw the line with the Seam painting tool (as an enforcer) before slicing, or "
    "use the project.")


BATTERY_LIGHT_NOTE = (
    "A bulb of the tree's own LED string needs no batteries at all. Battery lights trade "
    "brightness against running time, and bigger cells buy more of both. The brightest that "
    "lasts is a string of micro LEDs on thin wire with a 3 x AA battery box and a timer (6 "
    "hours on, 18 off): sellers give about 48 hours of continuous light per set of cells, more "
    "when the timer rests them - one to three weeks of evenings - and a version with a remote "
    "dimmer lasts longer turned down. Feed the lit end through the opening, fold it into a "
    "small bundle that hangs in the middle of the ball (an LED resting against a picture shows "
    "as a bright spot) and leave the box on a branch; one string can light two or three "
    "ornaments that hang close together. Coin cells are for short use: a fairy string on "
    "2 x CR2032 gives about a day, a puck in the cap's seat a few bright hours and then a long "
    "dim tail, an LED balloon light two or three hours. LED tea lights and clip-on tree candles "
    "run for a hundred hours and more on one cell, but only because they are dim - a fraction "
    "of a tree light.")


def _light_step(p: Params) -> str:
    """Step 2 of 'After printing': how the light goes in."""
    size = f"{p.hole_diameter:g} mm"
    if p.seat:
        return ("2. Switch the light on and drop it into the seat on top of the cap, lit side "
                "down.")
    if p.light_preset == "balloon":
        return (f"2. Push the LED balloon light into the {size} opening in the cap until its "
                f"ring rests on the cap.")
    if p.light_preset == "fairy":
        return f"2. Feed the fairy lights through the {size} opening in the cap."
    return f"2. Push a tree light through the {size} opening in the cap."


def _hang_step(p: Params) -> str:
    """Step 3 of 'After printing': how the ornament is hung."""
    if p.seat and p.hanger == "two":
        return ("3. Tie a ribbon through both loops so that it passes over the light, and hang "
                "the ornament by it.")
    if p.seat and p.hanger == "none":
        return "3. Stand it between branches with the cap upwards, so that the light stays in."
    if p.hanger != "none":
        return "3. Hang it by a ribbon or hook through the loop(s) on the cap."
    return "3. Let it sit on the light, or wedge it between branches."


FINE_NOZZLE_NOTE = (
    "Fit the 0.2 mm nozzle before you print and change the nozzle size on the printer's screen "
    "to match - the printer does not notice the swap by itself. The ready-to-print project "
    "already selects the 0.2 mm printer, filament and process presets. Lines are half as wide "
    "and layers half as tall as with the 0.4 mm nozzle, so the picture is finer and the print "
    "takes about three times as long. A hole that small clogs more easily: use plain white PLA "
    "(nothing with fibres, glitter, marble, wood or glow particles) and keep the filament clean "
    "and dry. The first layer is only 0.1 mm thick, so wipe the plate before you start; if it "
    "does not hold on the textured plate, makers report that a smooth plate does (choose that "
    "plate in Bambu Studio as well).")


def relief_can_hang(p: Params) -> bool:
    """True if these settings allow picture details that start in mid-air."""
    return p.relief_side != "split" and not p.safe_edges


def supports_note(p: Params) -> str:
    """What to tell the user about supports and slicer warnings for these settings."""
    return SUPPORTS_NOTE + ("\n\n" + UNSAFE_EDGES_NOTE if relief_can_hang(p) else "")


def _decoration(label: str, place: str, edge: str) -> str:
    """A choice of one of the decoration lists, said in passing: 'String of fairy
    lights' along the top -> 'string of fairy lights along the top'."""
    what = label.split(" (")[0].lower()                 # without the note some labels end in
    if edge and what.endswith(" in the corners"):
        return what.replace(" in the corners", f" in the {edge} corners")
    if "wreath" in what:                                # (its name already says where it goes)
        return what
    return f"{what} {place}"


def picture_lines(p: Params, lay: Layout) -> list[str]:
    """One line per picture: its photo, its lettering and its decorations."""
    from . import overlay

    panel = overlay.Panel.of(lay)
    out = []
    for k in range(p.n_images):
        slot = p.images[k]
        parts = [os.path.basename(slot.path) if slot.path else
                 ("no photo - a plain panel" if slot.decorated else "no photo yet")]
        text = overlay.lettering_for(slot, panel)
        if text is not None:
            parts.append(f"lettering \"{' / '.join(line for line in text.lines if line)}\" "
                         f"in {text.font}, letters {text.cap:.1f} mm tall")
        decorations = [_decoration(choices[key], place, edge) for place, edge, key, choices in
                       (("along the top", "top", slot.deco_top, DECO_TOP),
                        ("along the bottom", "bottom", slot.deco_bottom, DECO_BOTTOM),
                        ("all round", "", slot.deco_frame, DECO_FRAME),
                        ("scattered over it", "", slot.deco_scatter, DECO_SCATTER))
                       if key != "none"]
        if decorations:
            parts.append("decorations: " + ", ".join(decorations))
        out.append(f"{k + 1}. " + "; ".join(parts))
    return out


def ball_line(p: Params, lay: Layout) -> str:
    """One line for the print sheet about the decorations on the plain parts
    of the ball ("" if there are none): what is made, and what was asked for
    but finds no room."""
    from . import checks, overlay

    if not p.ball_decorated:
        return ""
    ball = overlay.ball_for(p, lay, 1.0)
    missing = set(ball.no_room) if ball is not None else set()
    asked = [name for name in ("ball_dividers", "ball_top", "ball_bottom", "ball_pole")
             if getattr(p, name) != "none"]
    made = [text for text, name in zip(checks.ball_parts(p), asked) if name not in missing]
    left = [text for text, name in zip(checks.ball_parts(p), asked) if name in missing]
    line = "The ball itself: " + (", ".join(made) if made else "nothing")
    if left:
        line += " (no room for: " + ", ".join(left) + ")"
    return line


def settings_sheet(p: Params, lay: Layout, stats: dict | None = None) -> str:
    """Plain-text print guide saved next to the exported model."""
    lines = [f"{APP_NAME} - print settings", "=" * 46, ""]
    if stats:
        sx, sy, sz = stats["size_mm"]
        hours = estimated_print_hours(p, stats["volume_cm3"])
        lines += [f"Size on the bed : {sx:.1f} x {sy:.1f} x {sz:.1f} mm",
                  f"Plastic         : about {stats['weight_g']:.0f} g of PLA "
                  f"({stats['volume_cm3']:.1f} cm3)",
                  f"Print time      : roughly {hours:.1f} h (the slicer gives the exact figure)",
                  f"Mesh            : {stats['triangles']:,} triangles", ""]
    lines += ["Pictures", "-" * 46]
    for entry in picture_lines(p, lay):
        lines += _wrap(entry, 78)
    ball = ball_line(p, lay)
    if ball:
        lines += _wrap(ball, 78)
    lines.append("")
    lines.append("Bambu Studio settings")
    lines.append("-" * 46)
    for setting, value, why in slicer_settings(p):
        lines.append(f"{setting:26s}: {value}" + (f"   ({why})" if why else ""))
    if p.nozzle == "0.2":
        lines += ["", "0.2 mm nozzle", "-" * 46] + _wrap(FINE_NOZZLE_NOTE, 78)
    lines += ["", "Supports", "-" * 46]
    for paragraph in supports_note(p).split("\n\n"):
        lines += _wrap(paragraph, 78) + [""]
    lines.pop()
    lines += ["", "Seam", "-" * 46] + _wrap(SEAM_NOTE, 78)
    lines += [
        "",
        "After printing",
        "-" * 46,
        "1. Let the bed cool, then lift the ornament off and peel away the brim.",
        _light_step(p),
        _hang_step(p),
        "",
        "Use LED lights only: incandescent bulbs get hot enough to soften PLA.",
        "",
    ]
    lines += ["Battery lights", "-" * 46] + _wrap(BATTERY_LIGHT_NOTE, 78) + [""]
    return "\n".join(lines)
