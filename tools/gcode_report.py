"""Summarise a sliced Bambu Studio file (.3mf with g-code inside, or .gcode): what
the printer will actually do, layer by layer.  Used to prove that an exported
ornament slices cleanly - no supports, bridges only where the design expects
them, seams at the back.

    python tools/gcode_report.py output/christmas_sphere_demo_bambu_A1.3mf
"""
from __future__ import annotations

import collections
import math
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithosphere.params import BED_SIZE_MM  # noqa: E402
from lithosphere.slicer import wall_loop_starts  # noqa: E402

RISKY_WORDS = ("bridge", "overhang", "floating", "support")


def read_gcode(path: str) -> str:
    if path.lower().endswith(".3mf"):
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".gcode"))
            if not names:
                raise ValueError("this 3mf contains no g-code (it was not sliced)")
            return zf.read(names[0]).decode("utf-8", "replace")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


_WORD = re.compile(r"([XYZEF])(-?\d*\.?\d+)")


def analyse(gcode: str) -> dict:
    """Walk the g-code once and collect per-layer / per-feature filament and
    the extent of the walls on the bed."""
    header: dict[str, str] = {}
    feature = "(start)"
    z = 0.0
    x = y = None
    layers: list[dict] = []
    current: dict | None = None
    per_feature: collections.Counter = collections.Counter()      # mm of filament per feature
    box = [1e9, 1e9, -1e9, -1e9]                                  # XY extent of all walls
    relative_e = True
    last_e = 0.0
    for line in gcode.splitlines():
        if line.startswith(";"):
            body = line[1:].strip()
            if body.startswith("FEATURE:"):
                feature = body.split(":", 1)[1].strip()
            elif body.startswith("Z_HEIGHT:"):
                z = float(body.split(":", 1)[1])
                current = {"z": z, "e": collections.Counter()}
                layers.append(current)
            elif ":" in body and len(header) < 400:
                for part in body.split(";"):        # "model printing time: ..; total estimated time: .."
                    if ":" in part:
                        key, value = part.split(":", 1)
                        header.setdefault(key.strip().lower(), value.strip())
            continue
        if line.startswith("M82"):
            relative_e = False
        elif line.startswith("M83"):
            relative_e = True
        elif line.startswith("G92"):
            m = re.search(r"E(-?\d*\.?\d+)", line)
            if m:
                last_e = float(m.group(1))
        if not line.startswith(("G0 ", "G1 ", "G2 ", "G3 ")):
            continue
        words = dict(_WORD.findall(line.split(";", 1)[0]))
        nx = float(words["X"]) if "X" in words else x
        ny = float(words["Y"]) if "Y" in words else y
        moving = "X" in words or "Y" in words
        extruding = False
        if "E" in words:
            e = float(words["E"])
            de = e if relative_e else e - last_e
            if not relative_e:
                last_e = e
            extruding = de > 0 and moving and current is not None
            if extruding:
                current["e"][feature] += de
                per_feature[feature] += de
        if feature in ("Outer wall", "Inner wall") and extruding:
            box[0], box[1] = min(box[0], nx), min(box[1], ny)
            box[2], box[3] = max(box[2], nx), max(box[3], ny)
        x, y = nx, ny
    return {"header": header, "layers": layers, "per_feature": per_feature, "box": box,
            "gcode": gcode}


def seam_stats(data: dict) -> dict:
    """Where the wall loops that go right round the part start, relative to the
    back of the printer (+Y).  Outer walls that do not go round it - the rims
    of dips in a picture - have no choice of where to start and are left out."""
    box = data["box"]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # Lithophane Sphere Studio stands the ornament's axis exactly mid-plate; picture relief
    # and hanger loops make the extent of the walls lopsided by a fraction of a millimetre.
    if abs(cx - BED_SIZE_MM / 2) < 1.0 and abs(cy - BED_SIZE_MM / 2) < 1.0:
        cx = cy = BED_SIZE_MM / 2
    starts = wall_loop_starts(data["gcode"], (cx, cy))
    offsets = [abs(math.degrees(math.atan2(x - cx, y - cy))) for _z, x, y in starts]
    sideways = [math.radians(off) * math.hypot(x - cx, y - cy)
                for off, (_z, x, y) in zip(offsets, starts)]
    n = len(offsets) or 1
    return {"loops": len(offsets), "centre": (cx, cy), "worst": max(offsets, default=0.0),
            "worst_mm": max(sideways, default=0.0),
            "within": {limit: 100.0 * sum(1 for o in offsets if o <= limit) / n
                       for limit in (3, 6, 12)}}


def report(path: str) -> dict:
    data = analyse(read_gcode(path))
    header, layers, per_feature = data["header"], data["layers"], data["per_feature"]
    total = sum(per_feature.values()) or 1.0
    print(f"File: {path}")
    for key in ("total estimated time", "model printing time", "total layer number",
                "total filament weight [g]", "total filament length [mm]", "max_z_height"):
        if key in header:
            print(f"  {key:28s}: {header[key]}")
    print(f"  layers found                : {len(layers)}"
          + (f" (z {layers[0]['z']:.2f} .. {layers[-1]['z']:.2f} mm)" if layers else ""))
    print("  filament by feature:")
    for name, mm in per_feature.most_common():
        print(f"    {name:24s} {mm:9.1f} mm  {100 * mm / total:5.1f} %")
    walls = sum(mm for name, mm in per_feature.items() if "wall" in name.lower())
    print(f"  walls in total              : {100 * walls / total:.1f} % of the filament")
    support = [f for f in per_feature if "support" in f.lower()]
    print(f"  supports generated          : {'YES - ' + ', '.join(support) if support else 'no'}")

    # where the slicer saw unsupported material: bridges, overhanging walls, floating shells
    risky: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for layer in layers:
        for name, mm in layer["e"].items():
            if any(word in name.lower() for word in RISKY_WORDS):
                risky[name].append((layer["z"], mm))
    if not risky:
        print("  unsupported material        : none flagged by the slicer")
    for name, hits in risky.items():
        zs = [z for z, _ in hits]
        mm = sum(m for _, m in hits)
        print(f"  {name:24s}: {len(hits):4d} layers, z {min(zs):.2f} .. {max(zs):.2f} mm, "
              f"{mm:.1f} mm of filament ({100 * mm / total:.2f} %)")
    empty = [layer["z"] for layer in layers if not layer["e"]]
    print(f"  layers with no extrusion    : {len(empty)}" + (f" (z = {empty[:8]})" if empty else ""))

    seams = seam_stats(data)
    cx, cy = seams["centre"]
    print(f"  wall loops right round the part: {seams['loops']}, object centre on the bed "
          f"({cx:.1f}, {cy:.1f})")
    for limit, percent in seams["within"].items():
        print(f"    starting within {limit:2d} deg of the back: {percent:5.1f} %")
    print(f"    furthest from the back      : {seams['worst']:.1f} deg "
          f"({seams['worst_mm']:.1f} mm sideways)")
    data.update(supports=support, risky=dict(risky), empty_layers=empty, seams=seams,
                wall_percent=100 * walls / total)
    return data


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    report(sys.argv[1])
