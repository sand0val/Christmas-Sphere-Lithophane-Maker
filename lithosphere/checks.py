"""Printability checks: will this ornament print well and look right when lit?"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry, overlay
from .params import (BALL_DIVIDERS, BALL_POLE, BALL_RING, BED_SIZE_MM, CLOSURE_ANGLE_DEG,
                     EDGE_RAMP_MIN_DEG, LIMITS, MAX_PRINT_HEIGHT_MM, OUTSIDE_RAMP_MIN_DEG,
                     PLA_DENSITY_G_CM3, PRINTER_NAME, SEAT_CLEARANCE_MM, TWO_LINES_NOZZLES, Layout,
                     Params, least_true_wall, line_width, mesh_pitch, nozzle_for,
                     recommended_layer_height, two_line_wall)
from .render import transmission

OK, WARN, ERROR = "ok", "warn", "error"


@dataclass
class Check:
    level: str       # "ok" | "warn" | "error"
    title: str
    detail: str


MAX_CLEAN_BRIDGE_MM = 45.0     # flat ceilings wider than this start to sag noticeably

# A stroke of lettering prints clearly when it is at least this many lines wide.
# (Measured on sliced ornaments: of a groove 0.7 mm wide a 0.4 mm nozzle lays
# down about two thirds of the depth, of one 0.5 mm wide about half.)  The
# letters themselves have to be tall enough for the gaps inside them - the eye
# of an "e" is narrower than its strokes in a bold font - to stay open.
CLEAR_STROKE_LINES = 1.6
CLEAR_LETTER_LINES = 7.0
# Above the middle of a picture the outside of the wall faces down on the
# printer and cannot carry a sudden edge (imaging.outside_first): part of every
# such edge goes to the inside of the wall, which looks blurred when lit.  The
# higher up, the more.  Lettering stays crisp there only if its strokes are at
# least this wide for every degree of latitude above the middle (read off lit
# renderings: 0.9 mm strokes are sharp at 12 degrees, soft at 25 and smeared at
# 37; 2.3 mm strokes are still good at 37).
SOFT_STROKE_MM_PER_DEG = 0.045


def lettering_checks(p: Params, lay: Layout, only: int | None = None) -> list["Check"]:
    """What to say about the lettering of each picture (or of picture `only`,
    counted from 0): too fine to print, in the part of the picture where it
    comes out soft, or fine."""
    out: list[Check] = []
    panel = overlay.Panel.of(lay)
    nozzle = nozzle_for(p)
    clear = CLEAR_STROKE_LINES * nozzle.line_width
    tall = CLEAR_LETTER_LINES * nozzle.line_width
    for k in range(p.n_images) if only is None else [only]:
        slot = p.images[k]
        text = overlay.lettering_for(slot, panel)
        if text is None:
            continue
        what = f"Picture {k + 1}: letters {text.cap:.1f} mm tall with strokes about {text.stroke:.1f} mm wide"
        if text.substituted:
            out.append(Check(WARN, "Font not found",
                             f"Picture {k + 1}: this computer has no font called "
                             f"\"{slot.text_font}\", so the lettering is set in {text.font} "
                             f"instead. Choose a font on the Decorate page to settle it."))
        if text.missing:
            several = len(text.missing) > 1
            out.append(Check(WARN, "Characters the font does not have",
                             f"Picture {k + 1}: {text.font} has no shape for "
                             f"{' '.join(text.missing)} - {'they print' if several else 'it prints'} "
                             f"as an empty box. Choose a font that has "
                             f"{'them' if several else 'it'} (Segoe UI Symbol has hearts, stars "
                             f"and snowflakes) or leave {'them' if several else 'it'} out."))
        # the part of the block that matters most sits a little above its middle
        high = math.degrees(text.latitude + 0.25 * text.height / text.radius)
        if text.stroke < clear - 1e-9 or text.cap < tall - 1e-9:
            other = "" if p.nozzle == "0.2" else ", or print with the 0.2 mm nozzle"
            fix = ("Make the lettering bigger, or break it into shorter lines so that each can "
                   "be larger" if text.cap < tall - 1e-9 else
                   "Pick a sturdier font or raise Thicken letters")
            out.append(Check(WARN, "Lettering too small to print clearly",
                             f"{what}. With the {nozzle.diameter:g} mm nozzle, letters want to "
                             f"be at least {tall:.1f} mm tall with strokes of {clear:.1f} mm or "
                             f"more; smaller ones run together. {fix}{other}."))
        elif (p.relief_side == "outside" and p.safe_edges
              and text.stroke < SOFT_STROKE_MM_PER_DEG * high - 1e-9):
            out.append(Check(WARN, "Lettering high up will look soft",
                             f"{what}, placed in the upper part of the picture. That half is "
                             f"printed with its outside facing down, where the printer cannot "
                             f"build a sudden edge, so letters this fine come out soft there "
                             f"(the Glowing view shows it). Move the lettering down - it is "
                             f"sharpest below the middle - or make it bigger and bolder."))
        else:
            out.append(Check(OK, "Lettering", what + "."))
    return out


# Decorations on the plain parts of the ball are the thinnest wall cut into the frame.  With
# less than this between the two they glow less than half as bright again as the frame.
FAINT_BALL_MM = 0.5
_BETWEEN = {"stars": "stars", "snowflakes": "snowflakes", "beads": "a string of beads",
            "candy": "candy-cane stripes", "fir": "a fir branch"}


def ball_parts(p: Params) -> list[str]:
    """What is asked for on the plain parts of the ball, in words: 'stars
    between the pictures', 'zigzag round the top' ..."""
    said = []
    for name, choices, where in (("ball_dividers", BALL_DIVIDERS, "between the pictures"),
                                 ("ball_top", BALL_RING, "round the top"),
                                 ("ball_bottom", BALL_RING, "round the bottom"),
                                 ("ball_pole", BALL_POLE, "underneath")):
        key = getattr(p, name)
        if key != "none":
            said.append(f"{choices[key].lower()} {where}")
    return said


def ball_checks(p: Params, lay: Layout) -> list["Check"]:
    """What to say about the decorations on the plain parts of the ball: one
    that finds no room, and what to change to make room for it; otherwise what
    there is and how it will look."""
    if not p.ball_decorated:
        return []
    out: list[Check] = []
    ball = overlay.ball_for(p, lay, 1.0)        # (drawn coarsely: only the room matters here)
    missing = set(ball.no_room) if ball is not None else set()
    radius = lay.R
    if "ball_dividers" in missing:
        what = _BETWEEN.get(p.ball_dividers, BALL_DIVIDERS[p.ball_dividers].lower())
        if lay.n_images == 1:
            out.append(Check(WARN, "Nothing between the pictures to decorate",
                             f"With one picture the only gap is the one at the back, and that "
                             f"stays plain: the printer starts every layer there. Choose two or "
                             f"more pictures on the Pictures page to have {what} between them."))
        else:
            need = math.degrees(overlay.divider_width_needed(p.ball_dividers) / radius)
            need = math.ceil(need * 2 - 1e-9) / 2
            have = math.degrees(2 * overlay.divider_half(lay))
            if need <= LIMITS["gap_deg"][1]:
                fix = (f"Set Gap between pictures on the Sphere page to {need:g} degrees or more, "
                       f"or choose something else.")
            else:
                fix = "Choose something else, or a bigger sphere."
            out.append(Check(WARN, "No room between the pictures",
                             f"The gap between the pictures is {have:g} degrees - "
                             f"{have / 360 * 2 * math.pi * radius:.1f} mm at its widest - which is "
                             f"too narrow for {what}, so that is left out. {fix}"))
    above, below, _closed = overlay.ball_strips(lay)
    for name, (lo, hi), where in (("ball_top", above, "top"), ("ball_bottom", below, "bottom")):
        if name not in missing:
            continue
        kind = getattr(p, name)
        if name == "ball_bottom" and not overlay.ring_below(p):
            out.append(Check(WARN, "No ring round the bottom with the picture on the inside",
                             f"With Inside chosen on the Sphere page, the strip below the pictures "
                             f"is where the inside of the wall runs in to the closed end, and a "
                             f"{BALL_RING[kind].lower()} could not be built there, so that is left "
                             f"out. Choose Outside or Both on the Sphere page to have it."))
            continue
        have = (hi - lo) * radius
        need = overlay.ring_height_needed(kind)
        # the picture height at which the strip would be tall enough
        other = lay.phi_t if name == "ball_top" else lay.phi_c
        height = math.floor(2 * math.degrees(other - need / radius) + 1e-9)
        if height >= LIMITS["band_height_deg"][0]:
            fix = (f"Set Picture height on the Sphere page to {height:.0f} degrees or less, or "
                   f"choose something else.")
        else:
            fix = "Choose something else, or a bigger sphere."
        out.append(Check(WARN, f"No room round the {where}",
                         f"Between the pictures and the {'cap' if where == 'top' else 'closed end'} "
                         f"there are {have:.1f} mm of plain ball, and a "
                         f"{BALL_RING[kind].lower()} needs {need:.1f} mm, so that is left out. "
                         f"{fix}"))
    made = [text for text, name in zip(ball_parts(p), [n for n in ("ball_dividers", "ball_top",
                                                                   "ball_bottom", "ball_pole")
                                                       if getattr(p, n) != "none"])
            if name not in missing]
    cut_in = [name for name in ("ball_dividers", "ball_top", "ball_bottom")
              if getattr(p, name) != "none" and name not in missing]
    if cut_in and abs(p.t_frame - p.t_min) < FAINT_BALL_MM - 1e-9:
        # (the sliders move in steps of 0.05 mm: name a thickness that is on one of them)
        need = math.ceil((p.t_min + FAINT_BALL_MM) * 20 - 1e-6) / 20
        out.append(Check(WARN, "Decorations on the ball will hardly show",
                         f"They are as thick as the thinnest wall ({p.t_min:.2f} mm), and the "
                         f"plain frame they are set in is {p.t_frame:.2f} mm - too close to that "
                         f"for them to stand out when the ornament is lit. Make the frame at least "
                         f"{need:.2f} mm thick on the Light & dark page."))
    if made:
        notes = []
        if p.ball_dividers != "none" and "ball_dividers" not in missing:
            notes.append("The divider at the back stays plain, so that the seam is hidden in it.")
        if p.relief_side == "inside" and cut_in:
            notes.append("With Inside chosen on the Sphere page these are formed on the inside "
                         "of the wall like the pictures: they show softly when the ornament is "
                         "lit and not at all unlit.")
        elif p.ball_top != "none" and "ball_top" not in missing:
            notes.append("What is round the top is formed on the inside of the wall: it shows "
                         "softly when the ornament is lit and not at all unlit.")
        if p.ball_pole != "none" and "ball_pole" not in missing:
            notes.append(f"The shape underneath stands {overlay.POLE_RAISE:g} mm proud of the ball.")
        out.append(Check(OK, "Decorations on the ball",
                         (made[0][:1].upper() + made[0][1:] + "".join(f", {m}" for m in made[1:])
                          + ". " + " ".join(notes)).strip()))
    return out


def analyze(p: Params, lay: Layout, volume_cm3: float | None = None,
            problems: list[str] | None = None) -> list[Check]:
    """Checks for the current design, most important first."""
    out: list[Check] = []
    layer = recommended_layer_height(p)

    for text in problems or []:
        out.append(Check(WARN, "Photo missing",
                         text + " A placeholder is shown instead. (A picture can also be "
                                "lettering only: type some on the Decorate page.)"))
    for text in lay.notes:
        out.append(Check(WARN, "Adjusted", text))

    # --- size ---------------------------------------------------------------
    if lay.height > MAX_PRINT_HEIGHT_MM - 2 or lay.width > BED_SIZE_MM - 10:
        out.append(Check(ERROR, "Too big", f"{lay.width:.0f} x {lay.height:.0f} mm does not fit "
                                           f"the {PRINTER_NAME}."))
    else:
        out.append(Check(OK, "Fits the printer",
                         f"{lay.width:.0f} mm wide, {lay.height:.0f} mm tall on the bed."))

    # --- walls --------------------------------------------------------------
    nozzle = nozzle_for(p)
    if least_true_wall(p) < TWO_LINES_NOZZLES * nozzle.diameter - 1e-9:
        thinnest = min(p.t_min, p.t_frame)
        other = "" if p.nozzle == "0.2" else " (or the 0.2 mm nozzle)"
        out.append(Check(WARN, "Very thin highlights",
                         f"The thinnest wall is {thinnest:.2f} mm. Two lines of the "
                         f"{nozzle.diameter:g} mm nozzle side by side need a little more, so the "
                         f"brightest areas print as a single line: weaker, and it may show gaps. "
                         f"{math.ceil(two_line_wall(p) * 20 - 1e-9) / 20:.2f} mm or more{other} "
                         f"keeps two lines everywhere."))
    else:
        out.append(Check(OK, "Wall thickness", f"Walls run from {p.t_min:.2f} mm (brightest) to "
                                               f"{p.t_max:.2f} mm (darkest)."))
    ratio = float(1.0 / transmission(np.array(p.t_max), p.t_min))
    if ratio < 3.0:
        out.append(Check(WARN, "Low contrast",
                         f"Bright areas will only be about {ratio:.1f}x brighter than dark ones. "
                         f"Increase the thickest wall or reduce the thinnest."))
    elif p.t_max > 3.4:
        out.append(Check(WARN, "Dark picture",
                         "Walls thicker than about 3.4 mm let very little light through; a small "
                         "tree light may look dim."))
    else:
        out.append(Check(OK, "Contrast when lit",
                         f"Bright areas glow roughly {ratio:.0f}x brighter than the darkest ones."))

    # --- picture relief -----------------------------------------------------
    # Lit from inside, relief on the outside of the wall looks sharp and relief
    # on the inside looks blurred (see render.soften_inside).  Printed cap-down,
    # the inside of the wall is a ceiling in the half printed last and the
    # outside is an underside in the half printed first.
    if p.relief_side == "outside" and p.safe_edges:
        out.append(Check(OK, "Sharp picture, nothing overhangs",
                         f"The picture is on the outside of the wall, where it looks sharpest "
                         f"when lit. In the half printed first the outside faces down: where an "
                         f"edge there is too sudden to print, the outside ramps at "
                         f"{OUTSIDE_RAMP_MIN_DEG:.0f} deg from flat (no flatter than the standard "
                         f"shoulder) and the inside makes up the difference, so the wall keeps "
                         f"the thickness the picture asks for, is nowhere thinner than "
                         f"{least_true_wall(p):.2f} mm across, and nothing starts in mid-air."))
    if p.relief_side == "split":
        out.append(Check(WARN, "Upper half of the pictures will look soft",
                         "With Both, the upper half of each picture is carried on the inside of "
                         "the wall. Lit, relief on the inside looks blurred - the light spreads "
                         "sideways in the plastic after passing it - so small details in that "
                         "half wash out. Outside on the Sphere page keeps the whole picture "
                         "sharp."))
    elif p.relief_side == "inside":
        eased = (f" In the lower half, sharp bright-above-dark edges are also eased to "
                 f"{EDGE_RAMP_MIN_DEG:.0f} deg from flat so that they can be printed."
                 if p.safe_edges else "")
        out.append(Check(WARN, "The pictures will look soft",
                         "With Inside the ornament is smooth, but relief on the inside of the "
                         "wall looks blurred when lit - the light spreads sideways in the "
                         "plastic after passing it - so small details wash out." + eased +
                         " Outside on the Sphere page keeps the picture sharp."))
    if p.relief_side != "split" and not p.safe_edges:
        where = "lower" if p.relief_side == "inside" else "upper"
        out.append(Check(WARN, "Details may print in mid-air",
                         f"Print-safe edges is off. In the {where} half of the pictures a sharp "
                         f"edge can hang below the wall it belongs to; it prints as a blob and "
                         f"Bambu Studio reports it as a floating region. Switch Print-safe edges "
                         f"back on (Light & dark page)."))

    # --- lettering and decorations --------------------------------------------
    out += lettering_checks(p, lay)
    out += ball_checks(p, lay)

    # --- seam ---------------------------------------------------------------
    # Every layer starts on a line marked down the divider at the back (see
    # geometry.seam_line_faces); the divider has to be wide enough to hold it.
    pitch = mesh_pitch(p)
    divider = math.degrees(lay.sector - 2 * lay.panel_half)
    if geometry.seam_fits_divider(lay, pitch):
        out.append(Check(OK, "Seam hidden",
                         "The Bambu Studio project marks a line down the middle of the divider "
                         "at the back, and every layer starts on it - never in a picture."))
    elif divider < 0.25:
        out.append(Check(WARN, "Seam between two pictures",
                         "With no gap between the pictures, the print seam runs down the border "
                         "between the two pictures at the back and shows as a faint line there."))
    else:
        need = math.ceil(2 * geometry.seam_min_gap_deg(lay, pitch) - 1e-9) / 2
        out.append(Check(WARN, "Seam touches the pictures",
                         f"A {divider:.1f} deg gap is narrower than the print seam needs, so the "
                         f"seam shows at the very edge of the pictures either side of the divider "
                         f"at the back. A gap of {need:g} deg or more hides it."))

    # --- overhangs ----------------------------------------------------------
    shallowest = min(p.flare_angle, CLOSURE_ANGLE_DEG)     # shoulder outside, cone inside
    step = layer * 1.0 / math.tan(math.radians(shallowest))
    if p.flare_angle < 28:
        out.append(Check(WARN, "Shallow shoulder",
                         f"A {p.flare_angle:.0f} deg shoulder is a steep overhang. Print at "
                         f"{layer:.2f} mm layers; the surface next to the cap may be slightly rough."))
    else:
        out.append(Check(OK, "No supports needed",
                         f"The shape's shallowest overhang is {shallowest:.0f} deg from flat: at "
                         f"{layer:.2f} mm layers each layer steps out {step:.2f} mm "
                         f"({step / line_width(p) * 100:.0f}% of a line)."))
    bridge = 2 * lay.r_f
    if bridge > MAX_CLEAN_BRIDGE_MM:
        out.append(Check(WARN, "Wide bridge at the closed end",
                         f"The inside of the closed end finishes with a {bridge:.0f} mm flat "
                         f"ceiling that is bridged over air. It will print, but may sag a little "
                         f"inside; a thicker frame makes it smaller."))
    else:
        how = (f"a {CLOSURE_ANGLE_DEG:.0f} deg cone and a {bridge:.0f} mm bridged ceiling"
               if bridge > 0.6 else f"a {CLOSURE_ANGLE_DEG:.0f} deg cone")
        out.append(Check(OK, "Closed end prints cleanly",
                         f"Outside it is a true sphere; inside it finishes with {how} instead of "
                         f"a dome printed over thin air."))
    if lay.roof < p.t_min - 0.1:
        out.append(Check(WARN, "Thin spot at the very bottom",
                         f"With a frame this thin, a ring at the closed end is only {lay.roof:.2f} mm "
                         f"thick - thinner than your thinnest wall, so it glows brighter than the "
                         f"pictures. A frame of 1.4 mm or more avoids it."))

    # --- stability ----------------------------------------------------------
    footprint = p.cap_diameter
    rim = (p.cap_diameter - p.seat_diameter) / 2 if p.seat else None
    if lay.height / footprint > 5.0:
        out.append(Check(WARN, "Narrow footprint",
                         f"Only a {footprint:.0f} mm cap holds a {lay.height:.0f} mm tall print on "
                         f"the moving bed. Use a wider cap, or a 8 mm brim and slower speeds."))
    elif rim is not None and rim < 1.8:
        out.append(Check(WARN, "Thin rim round the seat",
                         f"With the seat cut into it, only a {rim:.1f} mm rim of the cap stands on "
                         f"the bed and holds the light. Make the cap about "
                         f"{p.seat_diameter + 5:.0f} mm across if the sphere allows it."))
    elif rim is not None:
        out.append(Check(OK, "Bed adhesion", f"The {rim:.1f} mm rim of the {footprint:.0f} mm cap "
                                             f"on the bed plus a 5 mm brim is a stable base for "
                                             f"this height."))
    else:
        out.append(Check(OK, "Bed adhesion", f"{footprint:.0f} mm cap on the bed plus a 5 mm brim "
                                             f"is a stable base for this height."))

    # --- weight ---------------------------------------------------------------
    if volume_cm3 is not None:
        grams = volume_cm3 * PLA_DENSITY_G_CM3
        if grams > 75:
            out.append(Check(WARN, "Heavy ornament",
                             f"About {grams:.0f} g - too heavy for most branches. A smaller sphere "
                             f"or thinner walls and frame save weight."))
        elif grams > 40:
            out.append(Check(OK, "Weight", f"About {grams:.0f} g of PLA - hang it on a sturdy "
                                           f"branch, close to the trunk."))
        else:
            out.append(Check(OK, "Weight", f"About {grams:.0f} g of PLA."))
        from .export import estimated_print_hours
        hours = estimated_print_hours(p, volume_cm3)
        if p.nozzle == "0.2":
            out.append(Check(OK, "Print time",
                             f"Roughly {hours:.0f} hours with the 0.2 mm nozzle at {layer:.2f} mm "
                             f"layers - about three times as long as with the 0.4 mm nozzle, "
                             f"because each line holds a quarter of the plastic (the slicer gives "
                             f"the exact figure)."))
        else:
            out.append(Check(OK, "Print time", f"Roughly {hours:.1f} hours at {layer:.2f} mm "
                                               f"layers (the slicer gives the exact figure)."))

    # --- light / hanger ---------------------------------------------------------
    if p.seat:
        fits = p.seat_diameter - SEAT_CLEARANCE_MM
        out.append(Check(OK, "Seat for the light",
                         f"A light up to {fits:.1f} mm across drops {p.seat_depth:g} mm into the "
                         f"top of the cap, lit side down, and rests on a ledge round the "
                         f"{p.hole_diameter:g} mm opening. Measure your light before printing - "
                         f"makes differ by a millimetre or so."))
        if p.hanger == "one":
            out.append(Check(WARN, "One loop tilts the light",
                             "Hung by one loop the ornament leans, and the light in its seat "
                             "leans with it. Two loops hang it straight, and a ribbon tied "
                             "through both passes over the light and keeps it in."))
    if p.light_preset in ("c7", "c9", "c6"):
        out.append(Check(WARN, "Use LED bulbs",
                         "Incandescent C6/C7/C9 bulbs get hot enough to soften PLA. LED versions "
                         "are fine."))
    if p.hanger != "none" and p.hanger_hole < 2.5:
        out.append(Check(WARN, "Small hanger hole", "Holes under 2.5 mm tend to close up when "
                                                    "printed; a ribbon may not fit."))
    return out


def worst_level(checks: list[Check]) -> str:
    levels = {c.level for c in checks}
    return ERROR if ERROR in levels else WARN if WARN in levels else OK


# ---------------------------------------------------------------------------
# mesh verification (used after export builds and by the tests)
# ---------------------------------------------------------------------------
def verify_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Topological sanity of a triangle mesh.

    watertight: every edge is used by exactly two triangles, in opposite
    directions (closed, consistently oriented surface).
    """
    f = np.asarray(faces, dtype=np.int64)
    n = int(vertices.shape[0])
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fwd = directed[:, 0] * n + directed[:, 1]
    rev = directed[:, 1] * n + directed[:, 0]
    uniq, counts = np.unique(fwd, return_counts=True)
    duplicated = int((counts > 1).sum())
    unmatched = int((~np.isin(rev, uniq)).sum())
    a, b, c = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    return {
        "watertight": duplicated == 0 and unmatched == 0,
        "open_edges": unmatched,
        "non_manifold_edges": duplicated,
        "degenerate_faces": int((area2 < 1e-12).sum()),
        "volume_mm3": volume,
        "finite": bool(np.isfinite(vertices).all()),
    }


def hanging_tips(vertices: np.ndarray, faces: np.ndarray, min_depth: float = 1e-6) -> np.ndarray:
    """Places where the model would start in mid-air, printed as it stands
    (z up, resting on z = 0).  Returns their (x, y, z).

    These are the points a slicer reports as "floating regions": a vertex that
    is lower than everything it is joined to, on a part of the surface that
    faces down, and not on the bed.  Layer by layer the printer reaches such a
    point before anything connects it to the rest of the model."""
    f = np.asarray(faces, dtype=np.int64)
    z = np.asarray(vertices[:, 2], dtype=np.float64)
    lowest_neighbour = np.full(len(z), np.inf)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        np.minimum.at(lowest_neighbour, f[:, a], z[f[:, b]])
        np.minimum.at(lowest_neighbour, f[:, b], z[f[:, a]])
    tip = (z < lowest_neighbour - min_depth) & (z > z.min() + 1e-6)
    if not tip.any():
        return np.empty((0, 3))
    # Every triangle round such a point rises away from it, so together they
    # form a little cone standing on its tip.  Is the plastic inside that cone
    # (a tip hanging in the air) or outside it (the bottom of a dimple, which
    # is harmless)?  The triangles face outwards from the plastic, so measure
    # the cone against a point in the middle of its rim: the volume comes out
    # positive when the plastic is inside.
    v = np.asarray(vertices, dtype=np.float64)
    index = np.full(len(z), -1)
    index[tip] = np.arange(int(tip.sum()))
    apex, first, second = [], [], []
    for k in range(3):                      # each triangle touching a candidate, apex first
        sel = tip[f[:, k]]
        apex.append(f[sel, k])
        first.append(f[sel, (k + 1) % 3])
        second.append(f[sel, (k + 2) % 3])
    apex, first, second = (np.concatenate(x) for x in (apex, first, second))
    slot = index[apex]
    rim = np.zeros((int(tip.sum()), 3))
    count = np.zeros(int(tip.sum()))
    np.add.at(rim, slot, v[first])
    np.add.at(count, slot, 1.0)
    rim /= count[:, None]
    normal = np.cross(v[first] - v[apex], v[second] - v[apex])
    inside = np.zeros(int(tip.sum()))
    np.add.at(inside, slot, -np.einsum("ij,ij->i", normal, rim[slot] - v[apex]))
    return v[tip][inside > 0]
