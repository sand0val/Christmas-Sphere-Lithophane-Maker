"""Bambu Studio hand-off: open a model in the app, or slice it without the window.

Slicing uses the Bambu Studio command line through `bambu_cli` (which deals
with its quirks).  The model is passed already sitting in the middle of the
bed and auto-arrange is switched off, because the arranger is free to turn the
model - and the ornament must keep its orientation so that the layer seam
lands on the divider at the back.

The model goes to the slicer as a 3MF that marks a line down that divider as
a seam enforcer (see geometry.seam_line_faces); the project Bambu Studio
writes keeps the marking.  After slicing, seam_summary() reads the g-code to
see where the layers really start, and whether every layer of the pictures is
a closed ring of two wall loops (the outside and the inside of the wall).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np

from . import export
from .params import BED_SIZE_MM, Layout, Params, derive

_CANDIDATES = (
    r"C:\Program Files\Bambu Studio\bambu-studio.exe",
    r"C:\Program Files (x86)\Bambu Studio\bambu-studio.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Bambu Studio\bambu-studio.exe"),
)

# Warnings Bambu Studio always raises with these presets; they need no action.
EXPECTED_WARNINGS = {
    "bed_temperature_too_high_than_filament": "the stock textured-plate preset; harmless for PLA",
    "not_support_traditional_timelapse": "only matters if you record a timelapse",
}
# Warnings that mean the model itself should be changed, and what to change.
# "Floating regions" are parts that start in mid-air.  The design is built so
# that there are none; if Bambu Studio still finds some, it is right.
WARNING_ADVICE = {
    "floating regions": "a picture detail starts in mid-air and would print as a blob: switch "
                        "Print-safe edges on (Light & dark page)",
    "floating cantilever": "part of the model sticks out sideways with nothing under it",
}


def find_bambu_studio() -> str | None:
    """Path of the Bambu Studio executable, or None if it is not installed."""
    override = os.environ.get("BAMBU_STUDIO_EXE")
    if override and os.path.isfile(override):
        return override
    for path in _CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def can_slice() -> bool:
    return os.name == "nt" and find_bambu_studio() is not None


def open_in_bambu_studio(model_path: str) -> bool:
    """Open a model in the Bambu Studio window.  Returns False if it is not installed."""
    exe = find_bambu_studio()
    if exe is None:
        return False
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([exe, os.path.abspath(model_path)], creationflags=flags,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)
    return True


def project_path_for(model_path: str) -> str:
    """Name of the ready-to-print Bambu Studio project written next to a model."""
    return os.path.splitext(model_path)[0] + "_bambu_A1.3mf"


def format_duration(seconds: float | None) -> str:
    if not seconds:
        return "unknown"
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def split_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    """(expected, unexpected) slicer warnings.  Unexpected ones that are
    understood come with what to do about them."""
    expected, unexpected = [], []
    for text in warnings:
        note = next((why for key, why in EXPECTED_WARNINGS.items() if key in text), None)
        if note:
            expected.append(f"{text}  [{note}]")
            continue
        advice = next((what for key, what in WARNING_ADVICE.items() if key in text), None)
        unexpected.append(f"{text}  [{advice}]" if advice else text)
    return expected, unexpected


def slice_mesh(vertices: np.ndarray, faces: np.ndarray, out_path: str, p: Params,
               timeout_s: float = 1800.0, seam_faces: np.ndarray | None = None) -> dict:
    """Slice a print-orientation mesh for the A1 with the lithophane settings
    (and the presets of the nozzle chosen in p).

    Writes a Bambu Studio project to out_path: the model in the middle of the
    plate with printer, filament and every process setting already chosen (it
    opens on the Prepare tab, ready for "Slice plate").  seam_faces marks the
    triangles every layer should start on (Build.seam_faces); the project
    keeps that marking, so slicing it again in Bambu Studio gives the same seam.

    Returns the result of bambu_cli.slice_with_bambu (ok, error, print_time_s,
    filament_g, layer_count, support_used, warnings, ...) plus "seam": where
    the layers really start in the sliced file (see seam_summary), or None.
    """
    from . import bambu_cli

    exe = find_bambu_studio()
    if exe is None:
        return {"ok": False, "error": "Bambu Studio is not installed."}
    work = tempfile.mkdtemp(prefix="lsphere_")
    try:
        # Sit the model in the middle of the bed ourselves; see the module docstring.
        placed = np.array(vertices, dtype=np.float64, copy=True)
        placed[:, 0] += BED_SIZE_MM / 2
        placed[:, 1] += BED_SIZE_MM / 2
        # the object is listed in Bambu Studio under the model's own name
        name = os.path.splitext(os.path.basename(out_path))[0]
        for suffix in ("_bambu_A1", ".gcode"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        model = os.path.join(work, (name or "ornament") + ".3mf")
        export.write_3mf(model, placed, faces, name or "ornament", seam_faces)
        result = bambu_cli.slice_with_bambu(
            model, out_path, export.process_overrides(p), process=export.process_preset(p),
            machine=export.machine_preset(p), filament=export.filament_preset(p),
            exe=exe, arrange=False, timeout_s=timeout_s)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not result.get("ok") and result.get("stderr"):
        # the CLI's one-line error rarely says why; its stderr does
        result["error"] = f"{result.get('error')}  {str(result['stderr']).strip()[-400:]}"
    result["seam"] = None
    if result.get("ok"):
        try:
            from . import overlay
            lay = derive(p)
            result["seam"] = seam_summary(out_path, lay, overlay.ball_reach(p, lay))
        except Exception:               # a check, never a reason to fail the export
            result["seam"] = None
    return result


# ---------------------------------------------------------------------------
# where the layers start
# ---------------------------------------------------------------------------
_WORD = re.compile(r"([XYEIJ])(-?\d*\.?\d+)")
_MIN_LOOP_MM = 30.0     # anything shorter is at the very tip of the cap or the closed end
# Bambu Studio writes moves too short to need any plastic as "G1 X.. Y.. E0" in the middle of
# a wall (a thousandth of a millimetre; common with the fine lines of a 0.2 mm nozzle).  A
# move without plastic that is shorter than this is part of the wall, not a travel.
_STUTTER_MM = 0.05


def _move_points(code: str, x: float, y: float, nx: float, ny: float,
                 words: dict[str, str]) -> list[tuple[float, float]]:
    """Points along one move after its start: just the end for a straight
    line, a few points along the curve for an arc (G2 clockwise, G3 counter-
    clockwise, centre given by I / J relative to the start)."""
    if code not in "23" or ("I" not in words and "J" not in words):
        return [(nx, ny)]
    ox, oy = x + float(words.get("I", 0.0)), y + float(words.get("J", 0.0))
    radius = math.hypot(x - ox, y - oy)
    a0, a1 = math.atan2(y - oy, x - ox), math.atan2(ny - oy, nx - ox)
    sweep = (a1 - a0) % (2 * math.pi) if code == "3" else -((a0 - a1) % (2 * math.pi))
    if abs(sweep) < 1e-9:                       # start = end: a full circle
        sweep = 2 * math.pi if code == "3" else -2 * math.pi
    steps = max(1, math.ceil(abs(sweep) / (math.pi / 8)))
    points = [(ox + radius * math.cos(a0 + sweep * k / steps),
               oy + radius * math.sin(a0 + sweep * k / steps)) for k in range(1, steps)]
    return points + [(nx, ny)]


_JOIN_MM = 1.0          # two ends this close together are the same point of a loop


def _loops_of_layer(paths: list[list[float]]) -> list[list[list[float]]]:
    """The loops that go right round the ornament among one layer's outer-wall
    paths (each: z, start x, start y, length, end x, end y, turn).  A loop is
    usually one closed path; now and then Bambu Studio prints one in several
    passes - it leaves the loop to print a small separate ridge of the relief
    and comes back to finish it.  Passes that carry on where another stopped
    are put together again, in the order they were printed."""
    loops: list[list[list[float]]] = []
    used = [False] * len(paths)

    def apart(a: list[float], b: list[float]) -> float:    # from the end of a to the start of b
        return math.hypot(a[4] - b[1], a[5] - b[2])

    def complete(chain: list[int]) -> list[int] | None:
        """Carry a chain of passes on until it is back where it began, or give up."""
        if apart(paths[chain[-1]], paths[chain[0]]) < _JOIN_MM:
            return chain
        if len(chain) >= 8:
            return None
        follow = [j for j, other in enumerate(paths)
                  if not used[j] and j not in chain and apart(other, other) >= _JOIN_MM
                  and apart(paths[chain[-1]], other) < _JOIN_MM]
        # the pass that carries the loop on is long; a ridge that happens to start at the
        # same place is short: try the long one first
        for j in sorted(follow, key=lambda j: -paths[j][3]):
            done = complete(chain + [j])
            if done is not None:
                return done
        return None

    for i in range(len(paths)):
        if used[i]:
            continue
        chain = complete([i])
        if chain is None:
            continue
        pieces = [paths[j] for j in chain]
        if sum(q[3] for q in pieces) >= _MIN_LOOP_MM and abs(sum(q[6] for q in pieces)) > math.pi:
            for j in chain:
                used[j] = True
            loops.append(pieces)
    return loops


def wall_loops(gcode: str, centre: tuple[float, float] = (BED_SIZE_MM / 2, BED_SIZE_MM / 2)
               ) -> list[tuple[float, list[tuple[float, float]]]]:
    """(z, starts) of each outer-wall loop that goes right round the ornament -
    the outside and the inside of the wall alike.  starts holds the (x, y)
    where the loop starts - the seam - followed by the places where the printer
    took it up again, if it was printed in more than one pass.

    centre is where the ornament's axis stands on the bed.  Outer walls that
    do not go round it are left out: those are the rims of dips in a picture,
    which have to start somewhere on their own outline."""
    cx, cy = centre
    found: list[tuple[float, list[tuple[float, float]]]] = []
    layer: list[list[float]] = []
    outer = False
    z = 0.0
    x = y = None
    path: list[float] | None = None     # z, start x, start y, length, end x, end y, turn

    def close() -> None:
        nonlocal path
        if path is not None and path[3] > 0.0:
            layer.append(path)
        path = None

    def end_layer() -> None:
        close()
        for pieces in _loops_of_layer(layer):
            found.append((pieces[0][0], [(q[1], q[2]) for q in pieces]))
        layer.clear()

    for line in gcode.splitlines():
        if line.startswith(";"):
            if line.startswith("; FEATURE:"):
                close()
                outer = line[10:].strip() == "Outer wall"
            elif line.startswith("; Z_HEIGHT:"):
                end_layer()
                z = float(line[11:])
            continue
        if line[:1] != "G" or line[1:3] not in ("0 ", "1 ", "2 ", "3 "):
            continue
        words = dict(_WORD.findall(line.split(";", 1)[0]))
        if "X" not in words and "Y" not in words:
            continue                            # no sideways move: nothing starts or ends
        nx = float(words["X"]) if "X" in words else x
        ny = float(words["Y"]) if "Y" in words else y
        plastic = float(words.get("E", 0.0))
        stutter = (path is not None and plastic == 0.0 and x is not None and y is not None
                   and math.hypot(nx - x, ny - y) < _STUTTER_MM)
        if outer and x is not None and y is not None and (plastic > 0.0 or stutter):
            if path is None:
                path = [z, x, y, 0.0, x, y, 0.0]
            px, py = x, y
            for qx, qy in _move_points(line[1], x, y, nx, ny, words):
                path[3] += math.hypot(qx - px, qy - py)
                # the angle this step turns through seen from the axis: once round is 360 deg
                ax, ay, bx, by = px - cx, py - cy, qx - cx, qy - cy
                path[6] += math.atan2(ax * by - ay * bx, ax * bx + ay * by)
                px, py = qx, qy
            path[4], path[5] = nx, ny
        else:
            close()                             # a travel move ends the path being followed
        x, y = nx, ny
    end_layer()
    return found


def wall_loop_starts(gcode: str, centre: tuple[float, float] = (BED_SIZE_MM / 2, BED_SIZE_MM / 2)
                     ) -> list[tuple[float, float, float]]:
    """(z, x, y) where each outer-wall loop that goes right round the ornament
    starts - in other words where the seams are (see wall_loops)."""
    return [(z, starts[0][0], starts[0][1]) for z, starts in wall_loops(gcode, centre)]


def seam_summary(project_path: str, lay: Layout,
                 reach: tuple[float, float] | None = None) -> dict:
    """How well the seams of a sliced project hide on the divider at the back.

    reach: the latitudes between which the wall carries relief, when that is
    more than the pictures (overlay.ball_reach: a ball with a ring of
    decoration above or below them).

    Looks at every wall loop within the height of the pictures and returns
    loops (how many), divider_deg (width of the divider), stray (how many of
    the loops start off the divider), on_divider (the share that start on it,
    0..1), worst_deg and worst_mm (how far from the middle of the divider the
    furthest start is), and elsewhere: how many of the stray loops start
    nowhere near the divider (see FAR_FROM_DIVIDER_DEG).

    It also counts the layers: layers (how many lie in the height of the
    pictures) and broken (how many of them do not have two wall loops going
    right round - the outside and the inside of the wall).  A layer falls
    short when the wall is somewhere too thin for two lines side by side: the
    slicer then joins the two loops into one there, or leaves a gap.

    Now and then a loop is printed in two passes (see wall_loops).  The wall
    is whole, but the second pass starts somewhere of its own: resumed counts
    those starts that lie in a picture rather than on a divider."""
    with zipfile.ZipFile(project_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".gcode"))
        gcode = zf.read(names[0]).decode("utf-8", "replace") if names else ""
    centre = BED_SIZE_MM / 2
    lo, hi = reach if reach is not None else (-lay.band_half, lay.band_half)
    below, above = lay.R * math.sin(lo), lay.R * math.sin(hi)      # heights above the middle of the ball

    def carries_relief(z: float) -> bool:           # (the middle of the ball prints at z = z_top)
        return below < lay.z_top - z < above

    divider_half = math.degrees(lay.sector / 2 - lay.panel_half)
    sector = math.degrees(lay.sector)
    offsets, distances = [], []
    rounds: dict[float, int] = {}
    resumed = 0
    for z, starts in wall_loops(gcode):
        if not carries_relief(z):
            continue
        (x, y), again = starts[0], starts[1:]
        off = abs(math.degrees(math.atan2(x - centre, y - centre)))     # 0 at the back
        offsets.append(off)
        distances.append(math.radians(off) * math.hypot(x - centre, y - centre))
        rounds[z] = rounds.get(z, 0) + 1
        for ax, ay in again:                        # there is a divider every `sector` degrees
            turn = math.degrees(math.atan2(ax - centre, ay - centre))
            if abs((turn + sector / 2) % sector - sector / 2) > divider_half:
                resumed += 1
    heights = {float(text) for text in re.findall(r"^; Z_HEIGHT: *(-?\d*\.?\d+)", gcode, re.M)}
    in_band = [z for z in heights if carries_relief(z)]
    rings = {"layers": len(in_band), "broken": sum(1 for z in in_band if rounds.get(z, 0) < 2),
             "resumed": resumed}
    if not offsets:
        return {"loops": 0, "divider_deg": 2 * divider_half, "stray": 0, "elsewhere": 0,
                "on_divider": 0.0, "worst_deg": 0.0, "worst_mm": 0.0, **rings}
    stray = sum(1 for off in offsets if off > divider_half)
    elsewhere = sum(1 for off in offsets if off > divider_half + FAR_FROM_DIVIDER_DEG)
    return {"loops": len(offsets), "divider_deg": 2 * divider_half, "stray": stray,
            "elsewhere": elsewhere, "on_divider": 1.0 - stray / len(offsets),
            "worst_deg": max(offsets), "worst_mm": max(distances), **rings}


def ring_note(seam: dict | None) -> tuple[bool, str]:
    """(sound, sentence) about the wall loops of a seam_summary(): sound is
    False if some layers of the pictures are not a closed ring of two loops."""
    if not seam or not seam.get("loops") or not seam.get("broken"):
        return True, ""
    n, total = seam["broken"], seam.get("layers") or seam["broken"]
    return False, (f"In {n} of the {total} layers in the height of the pictures the wall is not "
                   f"printed as two loops right round: somewhere it is too thin for two lines "
                   f"side by side, and the slicer joins the outside and the inside loop there. "
                   f"It still prints, but that spot is thinner than the picture asks and may "
                   f"show as a bright dot. A thicker thinnest wall (Light & dark page) avoids it.")


# A seam that is not held by the divider starts just beside it.  A loop that
# starts further than this from the divider's edge is something else: Bambu
# Studio now and then gives a wall loop the starting point of a small separate
# outline next to it (the tip of a ridge of the relief that a layer cuts off as
# an island) - its own source notes that weakness of matching loops when the
# Arachne wall generator is used.  It needs steep, sudden relief and is rare:
# 1 to 3 loops in 1,000 to 2,000 with a test photo of pure black and white
# blocks, 2 in 1,022 with the four demo pictures (the fourth is lettering),
# none with the first three.
FAR_FROM_DIVIDER_DEG = 20.0
FEW_LOOPS_SHARE = 0.01


def seam_note(seam: dict | None) -> tuple[bool, str]:
    """(hidden, sentence) describing a seam_summary() for the user.  hidden is
    False only if wall loops start in a picture beside the divider although
    there is one, or if more than a few start somewhere else altogether."""
    if not seam or not seam.get("loops"):
        return True, ""
    if seam["divider_deg"] < 0.5:
        return True, ("There is no gap between the pictures, so the print seam runs along the "
                      "border between the two pictures at the back.")
    resumed = seam.get("resumed", 0)
    if not seam["stray"] and not resumed:
        return True, (f"The print seam runs down the middle of the divider at the back (never "
                      f"more than {max(0.1, seam['worst_mm']):.1f} mm from its centre line).")
    elsewhere = seam.get("elsewhere", 0)
    if elsewhere == seam["stray"]:
        marks = elsewhere + resumed
        few = marks <= FEW_LOOPS_SHARE * seam["loops"]
        one = marks == 1
        return few, (f"The print seam runs down the middle of the divider at the back, except "
                     f"for {marks} of the {seam['loops']} wall loops in the height of the "
                     f"pictures: Bambu Studio starts {'that one' if one else 'those'} inside a "
                     f"picture or a decoration instead - at a small separate ridge of the "
                     f"relief, or where it prints a loop in two passes. "
                     f"{'It leaves' if one else 'Each leaves'} a "
                     f"mark one layer tall. Very sudden edges cause this: lettering and "
                     f"decorations always have some, and in a photo more Smoothing (Light & "
                     f"dark page) makes them rarer. A few such marks do not show.")
    return False, (f"The print seam is not fully hidden: {seam['stray']} of the {seam['loops']} "
                   f"wall loops in the height of the pictures start in a picture beside the "
                   f"divider at the back (up to {seam['worst_mm']:.1f} mm from its centre line). "
                   f"A wider gap between the pictures hides it.")
