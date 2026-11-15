r"""Watertight, printable mesh of the ornament.

The shell is two surfaces of revolution (outer and inner) joined at the top
face of the cap.  Where there is a picture the wall thickness follows the
photo: the outside surface, the inside surface, or part of each is displaced
radially (see imaging.outside_first and imaging.relief_share; lit from within,
relief on the outside looks sharp and relief on the inside blurred, so the
outside carries all it can).  Above the pictures the sphere runs
tangentially into a short cone (the "shoulder") that meets a cylindrical cap,
like the metal cap of a glass bauble.  Printed cap-down, the cap gives a wide
flat footprint on the bed and no outside surface is shallower than the chosen
shoulder angle, so no supports are needed.

    hanging frame                         printed (flipped, cap on the bed)

        o[___]o   <- cap + hanger loops            .-'''-.
        .'   '.   <- shoulder                     /       \
      .'       '.                                |         |
     |           | <- pictures                    \       /
      '.       .'                                  '.   .'
        '-...-'                                   o[___]o   <- on the bed

The closed end (bottom when hanging, printed last) is a perfect sphere outside;
inside it finishes as a 30 degree cone with a small flat ceiling that the
printer bridges, instead of a dome that would have to be printed over thin air.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import imaging
from . import overlay
from . import params as _params
from . import render
from .params import (PLA_DENSITY_G_CM3, GeometryError, Layout, Params, derive, hanger_dims,
                     mesh_pitch)

COARSE_STEP = math.radians(2.0)      # row spacing where there is no detail
CAP_SHADE_MM = 1.5                   # extra "thickness" so the cap reads darker in the lit preview


@dataclass
class Body:
    """The shell without hanger loops, in the hanging frame."""

    vertices: np.ndarray                 # (n, 3) float64
    faces: np.ndarray                    # (m, 3) int32
    thickness: np.ndarray | None         # per-vertex wall thickness (NaN = inner surface)
    n_lon: int
    n_outer: int                         # number of vertices on the outer surface
    outer_rows: int = 0                  # rings on each surface (each surface also has a
    inner_rows: int = 0                  # single pole vertex in front of its rings)
    picture_mm3: float = 0.0             # plastic the pictures add to a plain frame-thick shell
    lit: np.ndarray | None = None        # per vertex: the thickness the wall LOOKS when lit
    #                                      (relief on the inside of the wall is blurred)


@dataclass
class Build:
    """A finished mesh."""

    vertices: np.ndarray                 # (n, 3) float64, mm
    faces: np.ndarray                    # (m, 3) int32
    layout: Layout
    orientation: str                     # "print" (cap on z = 0) or "hanging"
    stats: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)   # photo problems (missing files ...)
    seam_faces: np.ndarray | None = None  # per triangle: on the seam line (print orientation only)


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------
def longitude_count(R: float, pitch: float) -> int:
    n = int(round(2 * math.pi * R / pitch))
    return max(96, n + (n % 2))


def _relief_rows(lay: Layout, pitch: float, start: float) -> tuple[np.ndarray, slice]:
    """Latitudes of the rows of the detailed surface, and the slice that holds
    the picture band.  Rows are dense in the band and coarse elsewhere.

    start is the latitude of the first row: -pi/2 for a surface that runs down
    to a pole vertex (that row is then left out, the pole stands in for it), or
    the latitude where the inside leaves the sphere at the closed end."""
    phis, band, _fine = _relief_grid(lay, pitch, start)
    return phis, band


def _relief_grid(lay: Layout, pitch: float, start: float,
                 reach: tuple[float, float] | None = None) -> tuple[np.ndarray, slice, slice]:
    """_relief_rows() for a ball that carries relief beyond its pictures too.

    reach: the latitudes between which there is relief (overlay.ball_reach);
    the dense rows of the picture band carry on, at the same spacing, as far
    as that.  Returns (latitudes, the slice that holds the picture band, the
    slice that holds all the dense rows)."""
    b = lay.band_half
    nb = max(8, int(round(2 * b * lay.R / pitch)))
    dphi = 2 * b / nb
    lo, hi = reach if reach is not None else (-b, b)
    # dense rows past the pictures, but never as far as the rows that join the closed end
    # and the shoulder
    more_lo = min(max(0, math.ceil((-b - lo) / dphi - 1e-9)), max(0, int((-b - start) / dphi - 1.0)))
    more_hi = min(max(0, math.ceil((hi - b) / dphi - 1e-9)), max(0, int((lay.phi_t - b) / dphi - 1.0)))
    dense = -b + (np.arange(-more_lo, nb + more_hi) + 0.5) * dphi
    lo_edge, hi_edge = dense[0] - dphi, dense[-1] + dphi
    n_lo = max(1, math.ceil((lo_edge - start) / COARSE_STEP))
    lower = np.linspace(start, lo_edge, n_lo + 1)
    if start <= -math.pi / 2 + 1e-9:
        lower = lower[1:]
    n_hi = max(1, math.ceil((lay.phi_t - hi_edge) / COARSE_STEP))
    upper = np.linspace(hi_edge, lay.phi_t, n_hi + 1)
    first = len(lower)
    return (np.concatenate([lower, dense, upper]), slice(first + more_lo, first + more_lo + nb),
            slice(first, first + len(dense)))


def _rows_to_keep(phis: np.ndarray, displaced: np.ndarray) -> np.ndarray:
    """Which of the relief rows one surface needs.

    A surface keeps every row where it leaves its plain sphere (displaced) and
    the row either side of those; across the stretches in between, where it is
    exactly spherical, coarse rows are enough.  The first and last rows join
    the closed end and the shoulder and always stay."""
    keep = displaced.copy()
    keep[1:] |= displaced[:-1]
    keep[:-1] |= displaced[1:]
    keep[0] = keep[-1] = True
    last = 0
    for j in range(1, len(phis) - 1):
        if keep[j]:
            last = j
        elif phis[j + 1] - phis[last] > COARSE_STEP:
            keep[j] = True
            last = j
    return keep


def _surface_vertices(pole_z: float, pre_rings: list[tuple[float, float]], rho,
                      phis: np.ndarray, lam: np.ndarray,
                      post_rings: list[tuple[float, float]]) -> tuple[np.ndarray, int]:
    """Vertices of one surface, bottom to top: pole vertex, plain rings
    (radius, height), sphere rows (radial distance rho at latitudes phis), then
    more plain rings for the shoulder and cap.  Returns (vertices, ring count)."""
    cl, sl = np.cos(lam), np.sin(lam)

    def rings(items: list[tuple[float, float]]) -> np.ndarray:
        if not items:
            return np.empty((0, len(lam), 3))
        return np.stack([np.stack([r * cl, r * sl, np.full_like(cl, z)], axis=-1)
                         for r, z in items])

    cp, sp = np.cos(phis)[:, None], np.sin(phis)[:, None]
    rho = np.broadcast_to(np.asarray(rho, dtype=np.float64), (len(phis), len(lam)))
    rows = np.stack([rho * cp * cl, rho * cp * sl, rho * sp], axis=-1)
    grid = np.concatenate([rings(pre_rings), rows, rings(post_rings)], axis=0)
    verts = np.concatenate([np.array([[0.0, 0.0, pole_z]]), grid.reshape(-1, 3)])
    return verts, grid.shape[0]


_face_cache: dict[tuple[int, int, int, bool], np.ndarray] = {}


def _surface_faces(n_rows: int, n_lon: int, offset: int, outward: bool) -> np.ndarray:
    """Triangles of a pole-capped grid.  outward=True winds counter-clockwise
    seen from outside the sphere.  Results are cached (treat them as read-only)."""
    key = (n_rows, n_lon, offset, outward)
    hit = _face_cache.get(key)
    if hit is not None:
        return hit
    i = np.arange(n_lon)
    i1 = (i + 1) % n_lon
    fan = np.stack([np.full(n_lon, offset), offset + 1 + i1, offset + 1 + i], axis=1)
    j = np.arange(n_rows - 1)[:, None] * n_lon + offset + 1
    a, b = j + i[None, :], j + i1[None, :]
    c, d = b + n_lon, a + n_lon
    quads = np.concatenate([np.stack([a, b, c], axis=-1).reshape(-1, 3),
                            np.stack([a, c, d], axis=-1).reshape(-1, 3)])
    faces = np.concatenate([fan, quads]).astype(np.int32)
    if not outward:
        faces = np.ascontiguousarray(faces[:, ::-1])
    if len(_face_cache) > 12:
        _face_cache.clear()
    _face_cache[key] = faces
    return faces


def closure_thickness(lay: Layout, phis: np.ndarray) -> np.ndarray:
    """Approximate wall thickness at outer latitudes below the point where the
    inside leaves the sphere (the closed end), for shading the preview."""
    colat = phis + math.pi / 2
    r = lay.ro_frame * np.sin(colat)
    z_out = lay.ro_frame * np.cos(colat)
    theta_c = math.pi / 2 - lay.phi_c
    r_c, z_c = lay.ri_frame * math.sin(theta_c), lay.ri_frame * math.cos(theta_c)
    z_in = np.where(r <= lay.r_f, lay.z_f, z_c + (r_c - r) * math.tan(theta_c))
    return np.clip((z_out - z_in) * np.cos(colat), 0.3, None)


def edge_ramp_slope(lay: Layout) -> float:
    """Slope of the frame ramp for this layout (see params.edge_ramp_slope)."""
    return _params.edge_ramp_slope(lay.band_half)


def _ramp_frame_edge(T: np.ndarray, phis: np.ndarray, band: slice, lay: Layout, p: Params,
                     relief_inside: bool, pitch: float) -> None:
    """Replace the sudden step between a picture and the frame with a ramp on
    the one edge where that step would be printed as a long ledge over thin air.

    Printed cap-down, the inside of the wall above the equator is a ceiling and
    the outside below the equator is an underside.  A frame that is suddenly
    thicker than a bright picture edge (relief inside), or a dark picture edge
    that is suddenly thicker than the frame (relief outside), would stick out
    of that leaning surface as an unsupported shelf running along the picture.
    The picture is left alone; only the plain frame next to it is reshaped, so
    lit it shows as a soft fade of a few millimetres instead of a hard line.
    """
    from scipy import ndimage

    # most the wall may thicken from one row to the next one up on the printer
    # (the wall leans more the further the ramp runs, so the ramp gets gentler)
    radius = _params.relief_radius(p)
    side = "inside" if relief_inside else "outside"
    step = imaging.overhang_slope_limit(side, phis, radius) * np.diff(phis) * radius
    width = max(3, int(round(1.5 / pitch)) | 1)          # ~1.5 mm, odd
    sigma = max(0.5, 0.6 / pitch)
    if relief_inside:
        # the edge nearest the closed end; the frame beyond it must not thicken too fast
        edge = T[band.start]
        soft = ndimage.gaussian_filter1d(ndimage.minimum_filter1d(edge, width, mode="wrap"),
                                         sigma, mode="wrap")
        level = np.minimum(edge, soft)                   # never above the real edge
        for j in range(band.start - 1, 0, -1):           # row 0 joins the closing cone: keep it
            level = np.minimum(T[j], level + step[j])
            T[j] = level
    else:
        # the edge nearest the cap; the frame before it must already be thick enough
        last = band.stop - 1
        edge = T[last]
        soft = ndimage.gaussian_filter1d(ndimage.maximum_filter1d(edge, width, mode="wrap"),
                                         sigma, mode="wrap")
        level = np.maximum(edge, soft)                   # never below the real edge
        for j in range(band.stop, len(phis) - 1):        # the last row joins the shoulder: keep it
            level = np.maximum(T[j], level - step[j - 1])
            T[j] = level


def _decorate_plain_parts(T: np.ndarray, phis: np.ndarray, lam: np.ndarray, p: Params,
                          lay: Layout, pitch: float, outside_first: bool) -> np.ndarray | None:
    """Lay the decorations of the plain parts of the ball (overlay.ball_for)
    over the wall thickness T, in place.  No point of a picture is changed.

    What is drawn between the pictures and below them is treated like a
    picture: with the relief shared "outside first" what the printer can build
    is settled afterwards, and with the other settings print-safe edges are
    applied here, as imaging.thickness_band does for the pictures.

    What is drawn ABOVE the pictures is not put into T.  It is returned (the
    change of thickness it asks for, None if there is none) and goes on the
    inside of the wall, which is a floor up there and can take any shape; the
    outside leans over too far to carry it (see imaging.outside_first)."""
    plain = np.full_like(T, p.t_frame)
    overlay.apply_ball(plain, phis, lam, p, lay, pitch, parts=("dividers", "bottom"))
    drawn = np.abs(plain - p.t_frame) > 1e-9
    if drawn.any():
        if p.safe_edges and not outside_first:
            plain = imaging.limit_overhang_slope(plain, phis, _params.relief_radius(p),
                                                 p.relief_side, _params.relief_reach(p))
            drawn = np.abs(plain - p.t_frame) > 1e-9
            # a softened edge may not run on to the rows that join the closed end and the
            # shoulder, nor into a picture: there the picture decides
            drawn[0] = drawn[-1] = False
            in_band = np.abs(phis) < lay.band_half
            for center in lay.centers:
                cols = np.abs(imaging.wrap_angle(lam - center)) <= lay.panel_half + 1e-12
                drawn[np.ix_(in_band, cols)] = False
        T[drawn] = plain[drawn]
    above = np.full_like(T, p.t_frame)
    overlay.apply_ball(above, phis, lam, p, lay, pitch, parts=("top",))
    above -= p.t_frame
    return above if np.abs(above).max(initial=0.0) > 1e-9 else None


def lit_thickness(p: Params, lay: Layout, phis: np.ndarray, band: slice, out_shift: np.ndarray,
                  in_shift: np.ndarray) -> np.ndarray:
    """The thickness the wall LOOKS, lit from inside, on the relief rows.

    Relief on the outside shows exactly as it is.  Relief on the inside is
    softened first (render.soften_inside), because that is how it looks in a
    real print; the lit preview is coloured by this, not by the true
    thickness.  band: the slice of evenly spaced rows that carry relief - the
    picture band, and with it the strips above and below when those are
    decorated."""
    soft = in_shift
    if np.abs(in_shift[band]).max(initial=0.0) > 1e-9:
        rows = phis[band]
        row_mm = float(rows[-1] - rows[0]) / max(1, len(rows) - 1) * lay.R
        col_mm = 2 * math.pi / in_shift.shape[1] * lay.R * np.cos(rows)
        soft = in_shift.copy()
        soft[band] = render.soften_inside(in_shift[band], row_mm, col_mm)
    return p.t_frame + out_shift + soft


# ---------------------------------------------------------------------------
# body
# ---------------------------------------------------------------------------
def build_body(p: Params, lay: Layout, photos: list[imaging.Photo] | None, pitch: float,
               dense: bool = False) -> Body:
    """Build the shell.

    photos=None leaves the picture band plain (frame thickness everywhere).
    dense=True makes the outer surface as finely divided as the picture and
    records the wall thickness under every outer vertex (used to colour the
    lit preview)."""
    n_lon = longitude_count(lay.R, pitch)
    lam = np.arange(n_lon) * (2 * math.pi / n_lon)
    south = -math.pi / 2
    closed = -lay.phi_c                 # where the inside leaves the sphere at the closed end

    # Wall thickness T on the relief rows: every latitude where both surfaces
    # are spherical, from the closed end up to the shoulder.
    decorated = photos is not None and p.ball_decorated
    phis, band, fine = _relief_grid(lay, pitch, closed,
                                    overlay.ball_reach(p, lay) if decorated else None)
    T = np.full((len(phis), n_lon), p.t_frame, dtype=np.float64)
    outside_first = p.relief_side == "outside" and p.safe_edges
    above = None            # what is drawn on the strip above the pictures (it goes on the inside)
    if photos is not None:
        T[band] = imaging.thickness_band(p, lay, lam, phis[band], photos, pitch)
        if decorated:
            above = _decorate_plain_parts(T, phis, lam, p, lay, pitch, outside_first)
        if p.relief_side != "split" and not outside_first:
            _ramp_frame_edge(T, phis, band, lay, p, p.relief_side == "inside", pitch)

    # Share the relief between the two surfaces.  Where the wall is not frame
    # thickness, the difference goes outwards on the outside surface, inwards
    # on the inside surface, or partly each.
    extra = T - p.t_frame
    if outside_first:
        # as much as the printer can build on the outside, the rest inside
        out_shift, in_shift = imaging.outside_first(
            extra, phis, *_params.outside_first_radii(p), wall=p.t_frame,
            lon_step=2 * math.pi / n_lon, least=_params.least_true_wall(p), inside_extra=above)
        T = p.t_frame + out_shift + in_shift            # the wall that is really built
    else:
        share = imaging.relief_share(p.relief_side, phis)[:, None]
        out_shift = share * extra
        in_shift = (1.0 - share) * extra
        if above is not None:
            in_shift = in_shift + above
            T = T + above
    outer_all = lay.ro_frame + out_shift
    inner_all = lay.ri_frame - in_shift

    picture_mm3 = 0.0
    if photos is not None:
        # Volume the pictures (and what is drawn on the plain parts) add compared
        # with a plain shell: integrate the change of the displaced radii over the
        # rows that carry relief (r^3 / 3 per steradian).
        cubes = (outer_all[fine] ** 3 - inner_all[fine] ** 3
                 - (lay.ro_frame ** 3 - lay.ri_frame ** 3))
        d_phi = 2 * lay.band_half / (band.stop - band.start)
        solid_angle = np.cos(phis[fine])[:, None] * d_phi * (2 * math.pi / n_lon)
        picture_mm3 = float((cubes * solid_angle).sum() / 3.0)

    # Rows of each surface.  The outside also covers the closed end below the
    # relief rows; for the lit preview it keeps every row so that the wall
    # thickness is known under each of its vertices.
    n_cap = max(2, math.ceil((closed - south) / COARSE_STEP))
    cap_phis = np.linspace(south, closed, n_cap + 1)[1:-1]
    cap_rise, pole_rise = None, 0.0
    if decorated and p.ball_pole != "none":
        # a shape on the closed end: the outside stands proud of the sphere there, on
        # rows as close together as those of the pictures
        many = max(2, math.ceil((closed - south) * lay.R / pitch))
        rows = np.linspace(south, closed, many + 1)[:-1]
        rise = overlay.pole_relief(p, lay, rows, lam, pitch)
        if rise is not None:
            pole_rise = float(rise[0].mean())
            keep = np.ones(len(rows) - 1, dtype=bool) if dense else \
                _rows_to_keep(rows[1:], (rise[1:] > 1e-9).any(axis=1))
            cap_phis, cap_rise = rows[1:][keep], rise[1:][keep]
            d_cap = (closed - south) / many
            picture_mm3 += float((((lay.ro_frame + rise[1:]) ** 3 - lay.ro_frame ** 3)
                                  * np.cos(rows[1:])[:, None]).sum()
                                 * d_cap * (2 * math.pi / n_lon) / 3.0)
    cap_rho = np.full((len(cap_phis), n_lon), lay.ro_frame)
    if cap_rise is not None:
        cap_rho = cap_rho + cap_rise
    if dense:
        outer_keep = np.ones(len(phis), dtype=bool)
    else:
        outer_keep = _rows_to_keep(phis, (np.abs(out_shift) > 1e-9).any(axis=1))
    inner_keep = _rows_to_keep(phis, (np.abs(in_shift) > 1e-9).any(axis=1))
    outer_phis = np.concatenate([cap_phis, phis[outer_keep]])
    outer_rho = np.concatenate([cap_rho, outer_all[outer_keep]])
    inner_phis, inner_rho = phis[inner_keep], inner_all[inner_keep]
    outer_thickness = outer_lit = None
    if dense:
        cap_t = np.repeat(closure_thickness(lay, cap_phis)[:, None], n_lon, axis=1)
        if cap_rise is not None:
            cap_t = cap_t + cap_rise
        outer_thickness = np.concatenate([cap_t, T])
        outer_lit = np.concatenate([cap_t, lit_thickness(p, lay, phis, fine, out_shift, in_shift)])

    # Closed end, inside: cone -> flat bridged ceiling (see params.closure_profile).
    inner_pre = [(lay.r_f, -lay.z_f)] if lay.r_f > 0.3 else []
    # Shoulder cone and cap.  Outside: sphere -> cone -> cap wall.  Inside: sphere
    # -> cone -> cap inner wall -> (underside of the top plate) -> light opening,
    # which with a seat widens through a cone into the pocket the light sits in.
    outer_post = [(lay.rc, lay.z_oc), (lay.rc, lay.z_top)]
    inner_post = [(lay.rci, lay.z_ic)]
    if lay.cup:
        inner_post += [(lay.rci, lay.z_plate), (lay.rh, lay.z_plate)]
    if lay.seat:
        inner_post += [(lay.rh, lay.z_bore), (lay.r_seat, lay.z_seat), (lay.r_seat, lay.z_top)]
    else:
        inner_post.append((lay.rh, lay.z_top))

    outer_v, outer_rows = _surface_vertices(-lay.ro_frame - pole_rise, [], outer_rho, outer_phis,
                                            lam, outer_post)
    inner_v, inner_rows = _surface_vertices(-lay.z_f, inner_pre, inner_rho, inner_phis, lam,
                                            inner_post)
    n_outer = len(outer_v)
    faces = [_surface_faces(outer_rows, n_lon, 0, True),
             _surface_faces(inner_rows, n_lon, n_outer, False)]

    # Flat top of the cap joining the two surfaces (it rests on the bed when printing).
    i = np.arange(n_lon)
    i1 = (i + 1) % n_lon
    o = 1 + (outer_rows - 1) * n_lon
    q = n_outer + 1 + (inner_rows - 1) * n_lon
    faces.append(np.stack([o + i, o + i1, q + i1], axis=1).astype(np.int32))
    faces.append(np.stack([o + i, q + i1, q + i], axis=1).astype(np.int32))

    thickness = lit = None
    if dense:
        cap_shade = p.cap_wall + CAP_SHADE_MM
        inside = np.full(len(inner_v), np.nan)
        inside[-n_lon:] = cap_shade                     # top face of the cap
        pole_t = float(lay.ro_frame - lay.z_f) + pole_rise

        def per_vertex(outer_values: np.ndarray) -> np.ndarray:
            return np.concatenate([[pole_t], outer_values.ravel(),
                                   np.full(len(outer_post) * n_lon, cap_shade), inside])

        thickness, lit = per_vertex(outer_thickness), per_vertex(outer_lit)
    return Body(vertices=np.concatenate([outer_v, inner_v]), faces=np.concatenate(faces),
                thickness=thickness, n_lon=n_lon, n_outer=n_outer, outer_rows=outer_rows,
                inner_rows=inner_rows, picture_mm3=picture_mm3, lit=lit)


def body_normals(body: Body, vertices: np.ndarray | None = None) -> np.ndarray:
    """Smooth per-vertex normals from the grid structure (much faster than
    accumulating face normals).  Pass transformed vertices to get normals for
    another orientation; the transform must be a rotation."""
    v = body.vertices if vertices is None else vertices
    out = np.empty(v.shape, dtype=np.float32)

    def surface(first: int, rows: int, sign: float) -> None:
        grid = v[first + 1:first + 1 + rows * body.n_lon].reshape(rows, body.n_lon, 3)
        east = np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1)
        north = np.empty_like(grid)
        north[1:-1] = grid[2:] - grid[:-2]
        north[0] = grid[1] - grid[0]
        north[-1] = grid[-1] - grid[-2]
        n = np.cross(east, north) * sign
        length = np.linalg.norm(n, axis=2, keepdims=True)
        out[first + 1:first + 1 + rows * body.n_lon] = (n / np.where(length > 0, length, 1.0)
                                                         ).reshape(-1, 3)
        pole = n[0].sum(axis=0)             # the pole takes the average of its first ring
        out[first] = pole / (np.linalg.norm(pole) or 1.0)

    surface(0, body.outer_rows, 1.0)
    surface(body.n_outer, body.inner_rows, -1.0)
    return out


# ---------------------------------------------------------------------------
# hanger loops
# ---------------------------------------------------------------------------
def hanger_angles(p: Params) -> list[float]:
    """Directions (degrees of longitude) of the hanger loops."""
    if p.hanger == "two":
        return [0.0, 180.0]
    if p.hanger == "one":
        return [90.0]
    return []


def hanger_manifold(p: Params, lay: Layout, flush: bool = False):
    """Hanger loops as a manifold3d solid in the hanging frame, or None.

    Each loop is a flat tab that grows out of the cap wall at its top edge, so
    when printing it lies on the bed (no overhang, extra adhesion).  Normally
    the tab starts inside the cap wall so the two fuse into one solid;
    flush=True trims it at the cap surface instead (for drawing only).
    """
    angles = hanger_angles(p)
    if not angles:
        return None
    import manifold3d as m3

    d = hanger_dims(p, lay.rc)
    root = m3.CrossSection.circle(min(d["half_w"], lay.rc - 0.1), 64)
    eye = m3.CrossSection.circle(d["half_w"], 64).translate((d["cx"], 0.0))
    tab = m3.CrossSection.batch_hull([root, eye])
    tab = tab - m3.CrossSection.circle(d["hole_r"], 48).translate((d["cx"], 0.0))
    # start the tab inside the cap wall so it fuses with it but never pokes into the cap
    # (nor into the pocket of a seated light, which is cut into the top of that wall)
    hollow = max(lay.rci, lay.r_seat)
    start = lay.rc - 0.01 if flush else hollow + 0.4 * (lay.rc - hollow)
    tab = tab - m3.CrossSection.circle(start, 128)
    solid = m3.Manifold.extrude(tab, d["thickness"]).translate((0.0, 0.0, lay.z_top - d["thickness"]))
    parts = [solid.rotate((0.0, 0.0, ang)) for ang in angles]
    return parts[0] if len(parts) == 1 else m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def hanger_mesh(p: Params, lay: Layout) -> tuple[np.ndarray, np.ndarray] | None:
    """Hanger loops as plain arrays (for the preview)."""
    solid = hanger_manifold(p, lay, flush=True)
    if solid is None:
        return None
    mesh = solid.to_mesh()
    # copies, so that nothing keeps the manifold3d object alive after this call
    return (np.array(mesh.vert_properties, dtype=np.float64)[:, :3].copy(),
            np.array(mesh.tri_verts, dtype=np.int32))


def _union_with_hangers(body: Body, p: Params, lay: Layout) -> tuple[np.ndarray, np.ndarray]:
    solid = hanger_manifold(p, lay)
    if solid is None:
        return body.vertices, body.faces
    import manifold3d as m3

    mesh = m3.Mesh64(vert_properties=np.ascontiguousarray(body.vertices, dtype=np.float64),
                     tri_verts=np.ascontiguousarray(body.faces, dtype=np.uint64))
    shell = m3.Manifold(mesh)
    if shell.status() != m3.Error.NoError:
        raise GeometryError(f"Internal error: shell is not a valid solid ({shell.status()}).")
    joined = (shell + solid).to_mesh64()
    return (np.asarray(joined.vert_properties, dtype=np.float64)[:, :3],
            np.asarray(joined.tri_verts, dtype=np.int32))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def to_print_orientation(vertices: np.ndarray, lay: Layout) -> np.ndarray:
    """Turn the hanging-frame shape cap-down with the cap's top face on z = 0.

    This is a half-turn about the Y axis (a rotation, so nothing is mirrored)
    and it keeps +Y - where a divider sits - at the back of the printer."""
    out = np.empty_like(vertices)
    out[:, 0] = -vertices[:, 0]
    out[:, 1] = vertices[:, 1]
    out[:, 2] = lay.z_top - vertices[:, 2]
    return out


# Seam.  Bambu Studio's "Back" seam is only roughly at the back.  It starts
# from the point of each loop that is furthest back (largest Y) - which picture
# relief beside the divider can move off the divider - and then lets the seam
# drift sideways from layer to layer as long as it stays within 1.5 mm of that
# in Y, which on a ball this size is about 15 degrees either way.  Slices of
# this ornament had seams up to 14 degrees from the back, and where the seam
# settles changes from one slice to the next: no sensible divider is wide
# enough to be sure of holding it.  The exported 3MF therefore marks a narrow
# line down the middle of the back divider as a seam enforcer (what Bambu
# Studio's "Seam painting" tool does), which the slicer obeys before anything
# else.
SEAM_LINE_DEG = 1.0           # half-width of that line, in degrees of longitude
SEAM_LINE_REACH_MM = 0.42     # the slicer also takes points within one line width of it


def _seam_reach_deg(lay: Layout) -> float:
    """SEAM_LINE_REACH_MM as an angle, on the smallest layer that crosses a picture."""
    return math.degrees(SEAM_LINE_REACH_MM / (lay.ri_frame * math.cos(lay.band_half)))


def seam_line_half_deg(lay: Layout, pitch: float) -> float:
    """Half-width of the seam line for this layout, in degrees of longitude:
    a degree, less where the divider at the back is narrow, but never narrower
    than one column of the mesh (whole triangles are marked)."""
    column = 360.0 / longitude_count(lay.R, pitch)
    divider_half = math.degrees(lay.sector / 2 - lay.panel_half)
    return max(1.05 * column, min(SEAM_LINE_DEG, divider_half - _seam_reach_deg(lay)))


def seam_min_gap_deg(lay: Layout, pitch: float) -> float:
    """Narrowest gap between the pictures (degrees) that still holds the seam
    line together with the slicer's tolerance either side of it."""
    column = 360.0 / longitude_count(lay.R, pitch)
    return 2.0 * (1.05 * column + _seam_reach_deg(lay))


def seam_fits_divider(lay: Layout, pitch: float) -> bool:
    """True if the seam line, with the slicer's tolerance, lies wholly on the
    divider at the back - so that no layer can start in a picture."""
    divider_half = math.degrees(lay.sector / 2 - lay.panel_half)
    return seam_line_half_deg(lay, pitch) + _seam_reach_deg(lay) <= divider_half + 1e-9


def seam_line_faces(vertices: np.ndarray, faces: np.ndarray, half_deg: float) -> np.ndarray:
    """Which triangles of a print-orientation mesh lie wholly within half_deg
    of the back (+Y) of the ornament's axis: one True / False per triangle."""
    off = np.degrees(np.arctan2(vertices[:, 0], vertices[:, 1]))      # 0 at the back
    return (np.abs(off) <= half_deg)[np.asarray(faces)].all(axis=1)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume in mm^3 (positive for outward-facing triangles)."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def build_ornament(p: Params, pitch: float | None = None, orientation: str = "print") -> Build:
    """Build the complete printable ornament (shell + hanger loops, one solid)."""
    t0 = time.perf_counter()
    lay = derive(p)
    if pitch is None:
        pitch = mesh_pitch(p)
    photos, problems = imaging.photos_for(p)
    body = build_body(p, lay, photos, pitch, dense=False)
    vertices, faces = _union_with_hangers(body, p, lay)
    volume = mesh_volume(vertices, faces)
    seam_faces = None
    if orientation == "print":
        vertices = to_print_orientation(vertices, lay)
        seam_faces = seam_line_faces(vertices, faces, seam_line_half_deg(lay, pitch))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    stats = {
        "pitch_mm": pitch,
        "triangles": int(len(faces)),
        "vertices": int(len(vertices)),
        "volume_cm3": volume / 1000.0,
        "weight_g": volume / 1000.0 * PLA_DENSITY_G_CM3,
        "size_mm": tuple(float(x) for x in (hi - lo)),
        "build_seconds": time.perf_counter() - t0,
    }
    return Build(vertices=vertices, faces=faces, layout=lay, orientation=orientation,
                 stats=stats, problems=problems, seam_faces=seam_faces)


@dataclass
class Preview:
    """Meshes for the on-screen preview (hanging frame)."""

    body: Body
    hangers: tuple[np.ndarray, np.ndarray] | None
    layout: Layout
    problems: list[str]
    volume_cm3: float
    seconds: float


_frame_volume_cache: dict[tuple, float] = {}


def frame_volume(p: Params, lay: Layout) -> float:
    """Volume (mm^3) of the ornament with plain panels instead of pictures.

    Depends only on the shape settings, so it is cached; together with
    Body.picture_mm3 it gives the weight without measuring the big mesh on
    every slider movement."""
    key = (p.diameter, p.relief_side, p.t_frame, p.t_max if p.relief_side == "outside" else 0.0,
           p.flare_angle, p.hole_diameter, p.cap_diameter, p.cap_height, p.cap_wall, p.hanger,
           p.hanger_hole, (p.seat_diameter, p.seat_depth) if p.seat else None)
    hit = _frame_volume_cache.get(key)
    if hit is None:
        plain = build_body(p, lay, None, pitch=1.0)
        hit = mesh_volume(plain.vertices, plain.faces)
        hangers = hanger_mesh(p, lay)
        if hangers is not None:
            hit += mesh_volume(*hangers)
        if len(_frame_volume_cache) > 64:
            _frame_volume_cache.clear()
        _frame_volume_cache[key] = hit
    return hit


def preview_pitch(R: float, sharp: bool = False) -> float:
    """Sample spacing for the preview: quick while dragging, sharp once idle."""
    return max(0.2, 2 * math.pi * R / 1150) if sharp else max(0.3, 2 * math.pi * R / 720)


def build_preview(p: Params, pitch: float | None = None, sharp: bool = False) -> Preview:
    """Lower-resolution build for the interactive preview."""
    t0 = time.perf_counter()
    lay = derive(p)
    if pitch is None:
        pitch = preview_pitch(lay.R, sharp)
    photos, problems = imaging.photos_for(p)
    body = build_body(p, lay, photos, pitch, dense=True)
    hangers = hanger_mesh(p, lay)
    volume = frame_volume(p, lay) + body.picture_mm3
    return Preview(body=body, hangers=hangers, layout=lay, problems=problems,
                   volume_cm3=volume / 1000.0, seconds=time.perf_counter() - t0)
