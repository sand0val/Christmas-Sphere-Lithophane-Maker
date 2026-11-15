#!/usr/bin/env python3
"""Draw the four built-in festive sample images for the lithophane Christmas sphere.

Running this script with no arguments (re)creates, in ``../samples``:

    sample_tree.png      decorated Christmas tree with a star, in falling snow
    sample_snowman.png   friendly snowman with hat and scarf in a snowy night landscape
    sample_reindeer.png  stag silhouette against a big glowing moon and a starry sky
    sample_greeting.png  "Merry Christmas 2026" greeting card with holly and snowflakes

(Optionally pass one or more of ``tree snowman reindeer greeting`` to redraw only those.)

Every image is 1400 x 1400 px, 8-bit greyscale, and is drawn *for lithophanes*: think of
a backlit picture in which bright = thin wall = glowing light.

* light sources (star, moon, snow, lettering) are bright against darker surroundings;
* the whole tonal range is used, with smooth mid-tone gradients and shaded volumes;
* there are no large areas of pure black and no hairlines - nothing important is
  thinner than about 8 px (about 0.4 mm on a 75 mm wide print), snow is soft 10+ px dots;
* the subject sits inside the central ~80 % and the background simply runs off the
  edges, because the picture is wrapped on a sphere and its corners are trimmed away.

Only Pillow, numpy and scipy are needed.  The output is deterministic (fixed seeds).
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

SIZE = 1400          # final image size in pixels (square)
SS = 4               # supersampling factor used when rasterising hard-edged shapes
OUT_DIR = Path(__file__).resolve().parent.parent / "samples"
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

# Elegant bold serif italics first, then plainer bold serifs, then whatever Pillow has.
GREETING_FONTS = (
    "georgiaz.ttf", "georgiab.ttf", "timesbi.ttf", "timesbd.ttf", "palabi.ttf", "palab.ttf",
    "DejaVuSerif-BoldItalic.ttf", "DejaVuSerif-Bold.ttf", "LiberationSerif-BoldItalic.ttf",
)

# Pixel-centre coordinate grids shared by all the analytic shading code.
YY, XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
XX += 0.5
YY += 0.5


# --------------------------------------------------------------------------------------
# Small maths helpers
# --------------------------------------------------------------------------------------
def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    """0 at e0, 1 at e1 (e0 may be larger than e1), with soft shoulders."""
    t = np.clip((np.asarray(x, dtype=np.float32) - e0) / float(e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def unit(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def blur(a, sigma):
    """Gaussian blur; big radii use three box passes (fast, and just as smooth)."""
    if sigma <= 0:
        return a
    if sigma < 2.5:
        return ndi.gaussian_filter(a, sigma, mode="nearest")
    w = int(round(math.sqrt(4.0 * sigma * sigma + 1.0)))
    w += (w + 1) % 2
    for _ in range(3):
        a = ndi.uniform_filter(a, size=w, mode="nearest")
    return a


def shift(a, dx, dy):
    """Copy of ``a`` moved by whole pixels (dx right, dy down); vacated area is 0."""
    dx, dy = int(round(dx)), int(round(dy))
    out = np.zeros_like(a)
    h, w = a.shape
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = a[ys0:ys1, xs0:xs1]
    return out


def disc(r):
    """Boolean disc-shaped footprint of radius r (for morphology)."""
    k = int(math.ceil(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 0.25


def over(img, alpha, tone):
    """Paint ``tone`` (number or array) onto img through coverage ``alpha``, in place."""
    img += alpha * (tone - img)


def vignette(img, strength=0.25, inner=540.0, outer=1000.0):
    d = np.hypot(XX - SIZE / 2.0, YY - SIZE / 2.0)
    img *= 1.0 - strength * smoothstep(inner, outer, d)


def disc_alpha(cx, cy, r):
    """Antialiased coverage of a filled circle (full canvas)."""
    return np.clip(r - np.hypot(XX - cx, YY - cy) + 0.5, 0.0, 1.0)


def window(cx, cy, r):
    """Slices of the canvas that contain the circle (cx, cy, r)."""
    x0 = max(int(math.floor(cx - r)), 0)
    x1 = min(int(math.ceil(cx + r)) + 1, SIZE)
    y0 = max(int(math.floor(cy - r)), 0)
    y1 = min(int(math.ceil(cy + r)) + 1, SIZE)
    return slice(y0, y1), slice(x0, x1)


def smooth_noise(rs, cell):
    """Smooth random field in about [-1, 1] with blobs roughly ``cell`` px across."""
    n = max(4, int(math.ceil(SIZE / float(cell))) + 3)
    small = Image.fromarray(rs.rand(n, n).astype(np.float32))
    big = small.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return (np.asarray(big, dtype=np.float32) - 0.5) * 2.0


# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------
def spline(pts, closed=False, n=14):
    """Smooth Catmull-Rom curve through the points (any number of columns)."""
    p = np.asarray(pts, dtype=np.float64)
    if closed:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(p[-2][None, :])
    return np.vstack(out)


def bezier(p0, p1, p2, n=24):
    """Quadratic Bezier from p0 to p2 with control point p1."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(q, dtype=np.float64) for q in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t ** 2 * p2


def star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


# --------------------------------------------------------------------------------------
# Antialiased shape rasteriser
# --------------------------------------------------------------------------------------
class Mask:
    """Coverage mask built from simple shapes.

    Shapes are given in final-image pixel coordinates, rasterised SS times larger with
    Pillow and box-filtered down, so edges are properly antialiased.  ``bbox`` limits the
    working area (purely a speed-up for small shapes).
    """

    def __init__(self, bbox=None):
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, SIZE, SIZE
        else:
            x0 = min(max(0, int(math.floor(bbox[0]))), SIZE - 1)
            y0 = min(max(0, int(math.floor(bbox[1]))), SIZE - 1)
            x1 = max(min(SIZE, int(math.ceil(bbox[2]))), x0 + 1)
            y1 = max(min(SIZE, int(math.ceil(bbox[3]))), y0 + 1)
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.im = Image.new("L", ((x1 - x0) * SS, (y1 - y0) * SS), 0)
        self.d = ImageDraw.Draw(self.im)

    def _pt(self, x, y):
        return ((float(x) - self.x0) * SS, (float(y) - self.y0) * SS)

    def poly(self, pts, fill=255):
        self.d.polygon([self._pt(x, y) for x, y in pts], fill=fill)
        return self

    def rect(self, x0, y0, x1, y1, fill=255):
        return self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill)

    def circle(self, x, y, r, fill=255):
        ax, ay = self._pt(x - r, y - r)
        bx, by = self._pt(x + r, y + r)
        self.d.ellipse([ax, ay, bx, by], fill=fill)
        return self

    def ellipse(self, x, y, rx, ry, angle=0.0, fill=255):
        t = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
        ca, sa = math.cos(angle), math.sin(angle)
        px = x + rx * np.cos(t) * ca - ry * np.sin(t) * sa
        py = y + rx * np.cos(t) * sa + ry * np.sin(t) * ca
        return self.poly(list(zip(px, py)), fill)

    def stroke(self, pts, radius, fill=255):
        """Round-capped stroke along a polyline.  ``radius`` is a number, one value per
        point, or a pair (start, end) for a taper along the length."""
        pts = np.asarray(pts, dtype=np.float64)
        n = len(pts)
        if np.ndim(radius) == 0:
            rad = np.full(n, float(radius))
        else:
            radius = np.asarray(radius, dtype=np.float64)
            if len(radius) == n:
                rad = radius
            else:
                seg = np.hypot(*np.diff(pts, axis=0).T)
                s = np.concatenate([[0.0], np.cumsum(seg)])
                rad = lerp(radius[0], radius[1], s / max(s[-1], 1e-9))
        for i in range(n):
            self.circle(pts[i, 0], pts[i, 1], rad[i], fill)
            if i + 1 < n:
                dx, dy = pts[i + 1] - pts[i]
                length = math.hypot(dx, dy)
                if length < 1e-6:
                    continue
                nx, ny = -dy / length, dx / length
                self.poly([(pts[i, 0] + nx * rad[i], pts[i, 1] + ny * rad[i]),
                           (pts[i + 1, 0] + nx * rad[i + 1], pts[i + 1, 1] + ny * rad[i + 1]),
                           (pts[i + 1, 0] - nx * rad[i + 1], pts[i + 1, 1] - ny * rad[i + 1]),
                           (pts[i, 0] - nx * rad[i], pts[i, 1] - ny * rad[i])], fill)
        return self

    def array(self):
        small = np.asarray(self.im.reduce(SS), dtype=np.float32) / 255.0
        if small.shape == (SIZE, SIZE):
            return small
        out = np.zeros((SIZE, SIZE), np.float32)
        out[self.y0:self.y1, self.x0:self.x1] = small
        return out


# --------------------------------------------------------------------------------------
# Reusable picture elements
# --------------------------------------------------------------------------------------
def ball(img, cx, cy, r, hi=1.0, lo=0.55, light=(-0.45, -0.55, 0.70), clip=None):
    """Paint a shaded sphere: tone ``hi`` where it faces the light, ``lo`` in shade."""
    ys, xs = window(cx, cy, r + 2)
    nx = (XX[ys, xs] - cx) / r
    ny = (YY[ys, xs] - cy) / r
    d2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
    lam = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    tone = lo + (hi - lo) * lam
    a = np.clip((1.0 - np.sqrt(d2)) * r + 0.5, 0.0, 1.0)
    if clip is not None:
        a = a * clip[ys, xs]
    view = img[ys, xs]
    view += a * (tone - view)


def soft_dots(dots, softness=0.45):
    """Coverage map of soft round dots; ``dots`` holds (x, y, radius, strength)."""
    out = np.zeros((SIZE, SIZE), np.float32)
    for x, y, r, s in dots:
        ys, xs = window(x, y, r * (1 + softness) + 2)
        d = np.hypot(XX[ys, xs] - x, YY[ys, xs] - y)
        a = smoothstep(r * (1 + softness), r * (1 - softness), d) * s
        np.maximum(out[ys, xs], a, out=out[ys, xs])
    return out


def scatter(rs, n, min_dist, accept=None, margin=0.0, tries=6000):
    """Up to n random points, no two closer than min_dist (simple dart throwing)."""
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        x = rs.uniform(margin, SIZE - margin)
        y = rs.uniform(margin, SIZE - margin)
        if accept is not None and not accept(x, y):
            continue
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py in pts):
            pts.append((x, y))
    return pts


def pine_outline(cx, base_y, height, width, tiers=4):
    """Outline of a simple storybook fir: stacked saw-tooth tiers."""
    top = base_y - height
    right = []
    for k in range(tiers):
        f = (k + 1) / tiers
        y_tip = top + height * (0.16 + 0.84 * f)
        w_tip = 0.5 * width * (0.34 + 0.66 * f)
        right.append((w_tip, y_tip))
        if k < tiers - 1:
            right.append((w_tip * 0.50, y_tip - height * 0.035))
    pts = [(cx, top)]
    pts += [(cx + w, y) for w, y in right]
    pts += [(cx - w, y) for w, y in reversed(right)]
    return pts


def paint_firs(img, specs, base_fn, sink=22.0):
    """Fir silhouettes standing on the line base_fn(x).  specs = (x, height, width, tone),
    painted in the order given, so list the far (lighter) trees first."""
    for x, h, w, tone in specs:
        base_y = float(base_fn(x)) + sink
        m = Mask((x - w / 2 - 4, base_y - h - 4, x + w / 2 + 4, base_y + 4))
        m.poly(pine_outline(x, base_y, h, w))
        over(img, m.array(), tone * lerp(1.25, 0.85, smoothstep(base_y - h, base_y, YY)))


def load_font(size, candidates=GREETING_FONTS):
    """First available font from the candidate list, else Pillow's built-in font."""
    for name in candidates:
        for path in (FONT_DIR / name, Path(name)):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)      # Pillow >= 10.1: scalable built-in font
    except TypeError:
        return ImageFont.load_default()


def text_width(text, size):
    box = load_font(int(round(size))).getbbox(text)
    return float(box[2] - box[0])


def text_mask(text, cx, baseline, size, grow=0.0):
    """Coverage mask of one line of text centred on cx with the given baseline.
    ``grow`` emboldens every stroke by that many pixels on each side."""
    font = load_font(int(round(size * SS)))
    stroke = int(round(grow * SS))
    m = Mask()
    try:
        left, _, right, _ = font.getbbox(text, anchor="ls")
        m.d.text((cx * SS - (left + right) / 2.0, baseline * SS), text, font=font, fill=255,
                 anchor="ls", stroke_width=stroke, stroke_fill=255)
    except (ValueError, TypeError, AttributeError):   # bitmap fallback font: no anchors
        left, _, right, bottom = m.d.textbbox((0, 0), text, font=font)
        m.d.text((cx * SS - (left + right) / 2.0, baseline * SS - bottom), text, font=font,
                 fill=255, stroke_width=stroke, stroke_fill=255)
    return m.array()


def thin_fraction(mask, min_px=8.0):
    """Share of a mask's area lost when strokes thinner than min_px are removed."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        return 0.0
    b = mask[max(ys.min() - 8, 0):ys.max() + 9, max(xs.min() - 8, 0):xs.max() + 9] > 0.5
    opened = ndi.binary_opening(b, structure=disc(min_px / 2.0))
    return float((b & ~opened).sum()) / float(b.sum())


def lettering(lines, min_px=9.0, tolerance=0.01):
    """Masks for lines of text [(text, cx, baseline, size), ...] set in one weight.

    The lettering is emboldened in half-pixel steps until (almost) no stroke is thinner
    than min_px, so it stays printable whichever font ends up being used."""
    masks = []
    for grow in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        masks = [text_mask(text, cx, baseline, size, grow) for text, cx, baseline, size in lines]
        if max(thin_fraction(m, min_px) for m in masks) < tolerance:
            break
    return masks


# --------------------------------------------------------------------------------------
# 1. Christmas tree
# --------------------------------------------------------------------------------------
def tier_outline(cx, top, bottom, hw, lobes, sag, lift):
    """One tier of boughs: pointed top, gently concave sides, scalloped lower edge."""
    def base(x):                      # the lower edge sags in the middle (cone seen from the front)
        return bottom - lift * ((x - cx) / hw) ** 2

    height = bottom - top
    pts = list(bezier((cx, top), (cx + 0.40 * hw, top + 0.66 * height), (cx + hw, base(cx + hw)), 28))
    for k in range(lobes):
        xa = cx + hw - 2.0 * hw * k / lobes
        xb = cx + hw - 2.0 * hw * (k + 1) / lobes
        for u in np.linspace(0.0, 1.0, 18)[1:]:
            x = xa + (xb - xa) * u
            pts.append((x, base(x) + sag * math.sin(math.pi * u) ** 0.8))
    pts.extend(bezier((cx - hw, base(cx - hw)), (cx - 0.40 * hw, top + 0.66 * height), (cx, top), 28)[1:-1])
    return pts, base


def gift(img, x0, y0, w, h, body, ribbon, lid_h=28.0, lip=8.0):
    """A wrapped present with lid, ribbon and bow; (x0, y0) is the top-left of the lid."""
    bbox = (x0 - lip - 60, y0 - 70, x0 + w + lip + 60, y0 + h + 30)
    xc = x0 + w / 2.0
    # soft shadow on the snow
    sh = Mask(bbox).ellipse(xc + 14, y0 + h + 2, w * 0.62, 13).array()
    img *= 1.0 - 0.22 * blur(sh, 7)
    side = 1.10 - 0.26 * smoothstep(x0, x0 + w, XX)               # lit from the left
    box = Mask(bbox).rect(x0, y0 + lid_h - 2, x0 + w, y0 + h).array()
    over(img, box, body * side)
    under_lid = np.clip(1.0 - (YY - (y0 + lid_h)) / 16.0, 0.0, 1.0) * box
    img *= 1.0 - 0.22 * under_lid
    lid = Mask(bbox).rect(x0 - lip, y0, x0 + w + lip, y0 + lid_h).array()
    over(img, lid, np.clip(body * side * 1.22, 0.0, 1.0))
    band = Mask(bbox).rect(xc - 13, y0, xc + 13, y0 + h).array()
    over(img, band, ribbon)
    bow = Mask(bbox)
    bow.ellipse(xc - 30, y0 - 17, 35, 19, math.radians(24))
    bow.ellipse(xc + 30, y0 - 17, 35, 19, math.radians(-24))
    bow.circle(xc, y0 - 6, 14)
    bow = bow.array()
    img *= 1.0 - 0.25 * blur(shift(bow, 4, 6), 5) * (1.0 - bow)
    over(img, bow, ribbon)


def make_tree():
    rs = np.random.RandomState(20261)
    cx = 700.0
    sx, sy = cx, 242.0                                    # star centre

    # --- night sky, lighter towards the horizon, glowing around the star -------------
    img = lerp(0.13, 0.27, smoothstep(0.0, 1120.0, YY))
    ds = np.hypot(XX - sx, YY - sy)
    img += 0.32 * np.exp(-(ds / 170.0) ** 2)
    ax, ay = np.abs(XX - sx), np.abs(YY - sy)
    img += 0.15 * (np.exp(-(ax / 20.0) ** 1.6 - ay / 120.0) + np.exp(-(ay / 20.0) ** 1.6 - ax / 120.0))

    # --- distant snowy hills with a few small firs ------------------------------------
    def far(x):
        return 1000.0 + 24.0 * np.sin(x / 215.0 + 0.9) + 10.0 * np.sin(x / 84.0 + 2.2)

    yf = far(XX)
    over(img, np.clip(YY - yf + 0.5, 0.0, 1.0), lerp(0.42, 0.33, smoothstep(0.0, 110.0, YY - yf)))
    for x, h, w in ((112, 240, 134), (232, 168, 100), (1176, 182, 106), (1292, 252, 140)):
        base_y = float(far(x)) + 16.0
        fir = Mask((x - w, base_y - h - 4, x + w, base_y + 4)).poly(pine_outline(x, base_y, h, w)).array()
        tone = lerp(0.50, 0.38, smoothstep(x - 12.0, x + 12.0, XX))        # lit from the left
        over(img, fir, tone * lerp(1.0, 0.86, smoothstep(base_y - h, base_y, YY)))

    # --- foreground snow --------------------------------------------------------------
    def ground(x):
        return 1122.0 - 14.0 * np.exp(-((x - cx) / 400.0) ** 2) + 7.0 * np.sin(x / 150.0 + 1.0)

    yg = ground(XX)
    ga = np.clip(YY - yg + 0.5, 0.0, 1.0)
    gtone = lerp(0.90, 0.60, smoothstep(0.0, 290.0, YY - yg))
    gtone = gtone - 0.10 * smoothstep(240.0, 720.0, np.abs(XX - cx)) + 0.025 * smooth_noise(rs, 170)
    over(img, ga, gtone)
    shadow = blur(Mask().ellipse(cx + 24, 1138, 372, 30).array(), 15)
    img *= 1.0 - 0.30 * shadow * ga

    # --- trunk ------------------------------------------------------------------------
    trunk = Mask((cx - 40, 1020, cx + 40, 1150)).rect(cx - 31, 1025, cx + 31, 1140).array()
    over(img, trunk, lerp(0.30, 0.15, smoothstep(cx - 31, cx + 31, XX)))

    # --- the tree: four snow-edged tiers, drawn bottom-up -----------------------------
    tiers = [(275.0, 480.0, 152.0, 3), (378.0, 664.0, 230.0, 4),
             (522.0, 856.0, 304.0, 5), (690.0, 1056.0, 376.0, 6)]
    sag = 34.0
    tree = np.zeros_like(img)
    baubles, lights = [], []
    for idx in (3, 2, 1, 0):
        top, bottom, hw, lobes = tiers[idx]
        lift = 0.07 * hw
        pts, base = tier_outline(cx, top, bottom, hw, lobes, sag, lift)
        T = Mask().poly(pts).array()
        # shadow that this tier casts on the boughs below it
        img *= 1.0 - 0.50 * blur(shift(T, 0, 26), 13) * tree
        # snow lies along the scalloped edge; the foliage shows above it
        band = 30.0 + 0.02 * hw
        wave = 7.0 * np.sin((XX - cx) / (2.0 * hw / lobes) * 2.0 * math.pi * 1.37 + idx * 1.9)
        yb = bottom - lift * ((XX - cx) / hw) ** 2 - band + wave
        fol_cover = np.clip(yb - YY + 0.5, 0.0, 1.0)
        foliage = lerp(0.64, 0.43, smoothstep(top + 30.0, bottom - 20.0, YY))
        foliage = foliage * (1.0 - 0.18 * smoothstep(0.0, 1.0, np.abs(XX - (cx - 0.12 * hw)) / hw))
        snow = lerp(0.76, 0.98, smoothstep(0.50, 0.93, blur(T, 8)))
        over(img, T, lerp(snow, foliage, fol_cover))
        tree = np.maximum(tree, T)
        # baubles hang from the snowy bough tips (not on the lowest tier)
        if idx < 3:
            for j in range(lobes):
                xj = cx + hw - 2.0 * hw * (j + 0.5) / lobes
                r = 25.0 if (j + idx) % 2 else 21.0
                baubles.append((xj, float(base(xj)) + sag + r + 3.0, r))
        # a row of fairy lights across the foliage of the wider tiers
        if idx > 0:
            inside = blur(T * fol_cover, 9)
            n = lobes + 3
            for j in range(n):
                x = cx + hw * 0.80 * (2.0 * (j + 0.5) / n - 1.0)
                y = bottom - lift * ((x - cx) / hw) ** 2 - band - 40.0 + (7.0 if j % 2 else -7.0)
                if inside[int(y), int(x)] > 0.93:
                    lights.append((x, y))

    # --- star -------------------------------------------------------------------------
    spts = star_points(sx, sy, 100.0, 46.0)
    sbox = (sx - 110, sy - 110, sx + 110, sy + 110)
    over(img, Mask(sbox).poly(spts).array(), 0.86)
    facets = Mask(sbox)
    for i in range(0, 10, 2):
        facets.poly([(sx, sy), spts[i], spts[i + 1]])
    over(img, facets.array(), 1.0)

    # --- decorations ------------------------------------------------------------------
    glow = soft_dots([(x, y, 15.0, 1.0) for x, y in lights], softness=0.9)
    img += 0.22 * glow * (1.0 - img)
    over(img, soft_dots([(x, y, 8.5, 1.0) for x, y in lights], softness=0.25), 0.97)
    bshadow = Mask()
    for x, y, r in baubles:
        bshadow.circle(x + 6, y + 9, r + 2)
    img *= 1.0 - 0.35 * blur(bshadow.array(), 7)
    for i, (x, y, r) in enumerate(baubles):
        ball(img, x, y, r, hi=1.0, lo=0.50 if i % 2 else 0.38)

    # --- presents ---------------------------------------------------------------------
    gift(img, 432.0, 1094.0, 130.0, 114.0, body=0.40, ribbon=0.14)
    gift(img, 836.0, 1112.0, 158.0, 96.0, body=0.57, ribbon=0.22)

    # --- falling snow -----------------------------------------------------------------
    keep_out = blur(tree, 20)
    pts = scatter(rs, 88, 84.0, accept=lambda x, y: (y < 1085 and keep_out[int(y), int(x)] < 0.02
                                                     and math.hypot(x - sx, y - sy) > 165))
    dots = [(x, y, rs.uniform(5.5, 12.0), rs.uniform(0.50, 0.90)) for x, y in pts]
    over(img, soft_dots(dots), 1.0)

    vignette(img, 0.22)
    return img


# --------------------------------------------------------------------------------------
# 2. Snowman
# --------------------------------------------------------------------------------------
def ribbon_masks(center, half_w, stripe, bbox=None, phase=0.0, fringe=0.0):
    """Band of constant width along a centre line.  Returns (whole band, every other
    stripe).  ``fringe`` adds a zig-zag end of that length."""
    c = np.asarray(center, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
    tang = np.gradient(c, axis=0)
    tang /= np.hypot(tang[:, 0], tang[:, 1])[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    left, right = c + nrm * half_w, c - nrm * half_w
    whole, alt = Mask(bbox), Mask(bbox)
    for i in range(len(c) - 1):
        quad = [left[i], left[i + 1], right[i + 1], right[i]]
        whole.poly(quad)
        if int((0.5 * (s[i] + s[i + 1]) + phase) // stripe) % 2 == 0:
            alt.poly(quad)
    if fringe > 0:
        teeth = 3
        for k in range(teeth):
            a = left[-1] + (right[-1] - left[-1]) * (k / teeth)
            b = left[-1] + (right[-1] - left[-1]) * ((k + 1) / teeth)
            whole.poly([a, b, (a + b) / 2.0 + tang[-1] * fringe])
    return whole.array(), alt.array()


def make_snowman():
    rs = np.random.RandomState(20262)
    cx = 705.0
    light = unit(-0.50, -0.58, 0.64)                       # moonlight from the upper left
    coal = 0.07

    # --- sky and crescent moon --------------------------------------------------------
    img = lerp(0.17, 0.42, smoothstep(-60.0, 840.0, YY))
    mx, my, mr = 338.0, 292.0, 98.0
    dm = np.hypot(XX - mx, YY - my)
    img += 0.20 * np.exp(-dm / 185.0)
    mbox = (mx - mr - 6, my - mr - 6, mx + mr + 6, my + mr + 6)
    crescent = Mask(mbox).circle(mx, my, mr).circle(mx + 42, my - 25, mr * 0.86, fill=0).array()
    img += 0.30 * blur(crescent, 16) * (1.0 - img)
    over(img, crescent, lerp(1.0, 0.90, smoothstep(my - mr, my + mr, YY)))

    # --- dark wooded hills on the horizon, with firs ----------------------------------
    def far(x):
        return 812.0 + 24.0 * np.sin(x / 250.0 + 0.5) + 9.0 * np.sin(x / 90.0 + 2.0)

    yf = far(XX)
    over(img, np.clip(YY - yf + 0.5, 0.0, 1.0), lerp(0.24, 0.30, smoothstep(0.0, 130.0, YY - yf)))
    paint_firs(img, [(40, 250, 150, 0.23), (268, 230, 136, 0.23), (1040, 170, 104, 0.23), (1262, 360, 190, 0.23),
                     (150, 330, 180, 0.14), (372, 150, 96, 0.14), (1142, 290, 164, 0.14), (1372, 240, 150, 0.14)],
               far)

    # --- snow field -------------------------------------------------------------------
    def field(x):
        return 938.0 + 20.0 * np.sin(x / 300.0 + 2.6) + 8.0 * np.sin(x / 110.0 + 0.4)

    ys_ = field(XX)
    fa = np.clip(YY - ys_ + 0.5, 0.0, 1.0)
    depth = YY - ys_
    ftone = lerp(0.66, 0.86, smoothstep(0.0, 190.0, depth)) - 0.16 * smoothstep(230.0, 470.0, depth)
    ftone = ftone - 0.08 * smoothstep(300.0, 760.0, np.abs(XX - cx)) + 0.025 * smooth_noise(rs, 190)
    over(img, fa, ftone)

    # --- the snowman's body -----------------------------------------------------------
    bx, by, br = cx, 985.0, 238.0          # bottom ball
    mxb, myb, mrb = cx, 700.0, 172.0       # middle ball
    hxh, hyh, hrh = cx, 455.0, 125.0       # head
    body = np.maximum(np.maximum(disc_alpha(bx, by, br), disc_alpha(mxb, myb, mrb)),
                      disc_alpha(hxh, hyh, hrh))
    img *= 1.0 - 0.20 * blur(body, 24) * (1.0 - 0.6 * fa)   # soft halo so the figure separates
    cast = blur(Mask().ellipse(cx + 190, 1196, 320, 42, math.radians(2)).array(), 16)
    img *= 1.0 - 0.32 * cast * fa
    # the ball sits a little way down in the snow
    snow_line = 1196.0 + 5.0 * np.sin(XX / 60.0)
    above_snow = 1.0 - smoothstep(-5.0, 5.0, YY - snow_line)

    ball(img, bx, by, br, hi=1.0, lo=0.50, light=light, clip=above_snow)
    img *= 1.0 - 0.30 * blur(disc_alpha(mxb + 16, myb + 34, mrb), 18) * disc_alpha(bx, by, br)
    ball(img, mxb, myb, mrb, hi=1.0, lo=0.52, light=light)
    img *= 1.0 - 0.28 * blur(disc_alpha(hxh + 12, hyh + 30, hrh), 16) * disc_alpha(mxb, myb, mrb)

    # --- twig arms --------------------------------------------------------------------
    arms = Mask()
    for sgn, pts, fingers in (
            (-1, [(cx - 148, 676), (cx - 240, 622), (cx - 330, 548)],
             [(-0.50, -0.87, 60), (-0.97, -0.26, 54), (0.12, -0.99, 50)]),
            (1, [(cx + 150, 676), (cx + 250, 632), (cx + 338, 590)],
             [(0.55, -0.83, 60), (0.99, -0.10, 54), (0.80, 0.60, 44)])):
        arms.stroke(spline(pts, n=12), (10.0, 7.5))
        hx_, hy_ = pts[-1]
        for ux, uy, ln in fingers:
            arms.stroke([(hx_ - ux * 4, hy_ - uy * 4), (hx_ + ux * ln, hy_ + uy * ln)], (6.5, 4.8))
    over(img, arms.array(), 0.10)

    # --- head -------------------------------------------------------------------------
    ball(img, hxh, hyh, hrh, hi=1.0, lo=0.54, light=light)

    # --- scarf: two hanging tails, the neck band and a knot -----------------------------
    sbox = (cx - 180, 500, cx + 230, 870)
    dark_s, light_s = 0.26, 0.60

    def paint_band(whole, alt, shade=1.0):
        over(img, whole, dark_s * shade)
        over(img, alt, light_s * shade)

    # the neck band is a crescent (a ring seen from the front); its stripes fan out
    band = Mask(sbox).poly(list(bezier((cx - 118, 532), (cx, 602), (cx + 118, 532), 40))
                           + list(bezier((cx + 124, 545), (cx, 722), (cx - 124, 545), 40))).array()
    band = band * smoothstep(109.0, 105.0, np.abs(XX - cx))            # blunt ends, just proud of the neck
    fan = np.degrees(np.arctan2(XX - cx, YY - 330.0))
    stripe_px = np.radians(8.8) * np.hypot(XX - cx, YY - 330.0)
    band_alt = band * np.clip(0.5 + np.sin(np.pi * (fan + 4.4) / 8.8) * stripe_px / np.pi, 0.0, 1.0)
    img *= 1.0 - 0.26 * blur(shift(band, 3, 13), 8)                    # its shadow on the body
    tail_b = spline([(cx + 62, 606), (cx + 50, 668), (cx + 44, 734)], n=24)
    w, a = ribbon_masks(tail_b, 26, 34, sbox, phase=10, fringe=17)
    paint_band(w, a, 0.84)
    tail_a = spline([(cx + 86, 600), (cx + 106, 672), (cx + 114, 748), (cx + 138, 818)], n=24)
    w, a = ribbon_masks(tail_a, 30, 34, sbox, fringe=19)
    img *= 1.0 - 0.22 * blur(shift(w, 7, 7), 6) * (1.0 - w)
    paint_band(w, a)
    fold = 1.0 - 0.30 * ((XX - cx) / 135.0) ** 2
    over(img, band, dark_s * fold)
    over(img, band_alt, light_s * fold)
    knot = Mask(sbox).ellipse(cx + 80, 604, 36, 25, math.radians(-24)).array()
    img *= 1.0 - 0.25 * blur(shift(knot, 3, 8), 5) * (1.0 - knot)
    over(img, knot, lerp(0.38, 0.22, smoothstep(585.0, 630.0, YY)))

    # --- face -------------------------------------------------------------------------
    nbx, nby, ntx, nty = cx - 12.0, 466.0, cx - 126.0, 491.0
    nlen = math.hypot(ntx - nbx, nty - nby)
    ux, uy = (ntx - nbx) / nlen, (nty - nby) / nlen
    nose = Mask((cx - 160, 420, cx + 30, 530)).stroke([(nbx, nby), (ntx, nty)], (23.0, 5.0)).array()
    img *= 1.0 - 0.22 * blur(shift(nose, 6, 12), 6) * (1.0 - nose)
    along = ((XX - nbx) * ux + (YY - nby) * uy) / nlen
    across = (-(XX - nbx) * uy + (YY - nby) * ux)
    over(img, nose, lerp(0.52, 0.36, np.clip(along, 0.0, 1.0)) - 0.10 * smoothstep(-8.0, 10.0, across))
    face = Mask((cx - 130, 380, cx + 130, 570))
    face.circle(cx - 46, 428, 15).circle(cx + 32, 430, 15)
    for k in range(5):
        a = math.radians(52 + 19 * k)
        face.circle(cx - 6 + 80 * math.cos(a), 452 + 80 * math.sin(a), 9.5)
    over(img, face.array(), coal)
    buttons = Mask((cx - 40, 640, cx + 40, 840))
    for yb_ in (674, 740, 806):
        buttons.circle(cx - 2, yb_, 16)
    over(img, buttons.array(), coal)

    # --- top hat with a snow cap --------------------------------------------------------
    hx0, hy0, tilt = cx - 6.0, 362.0, math.radians(-9.0)
    ct, st = math.cos(tilt), math.sin(tilt)

    def H(u, v):
        return (hx0 + u * ct - v * st, hy0 + u * st + v * ct)

    U = (XX - hx0) * ct + (YY - hy0) * st                  # hat-local coordinates
    V = -(XX - hx0) * st + (YY - hy0) * ct
    hbox = (cx - 190, 150, cx + 180, 420)
    us = np.linspace(-1.0, 1.0, 41)
    crown_pts = [H(93 * u, 18 * math.sqrt(max(0.0, 1 - u * u))) for u in us]
    crown_pts += [H(101 * math.cos(t), -160 - 20 * math.sin(t)) for t in np.linspace(0, math.pi, 41)]
    crown = Mask(hbox).poly(crown_pts).array()
    sheen = 0.075 + 0.12 * np.exp(-((U + 42.0) / 34.0) ** 2)
    over(img, crown, sheen)
    band_pts = [H(94 * u, -20 + 17 * math.sqrt(max(0.0, 1 - u * u))) for u in us]
    band_pts += [H(96 * u, -62 + 17 * math.sqrt(max(0.0, 1 - u * u))) for u in us[::-1]]
    over(img, Mask(hbox).poly(band_pts).array() * crown, 0.50 + 0.22 * np.exp(-((U + 42.0) / 40.0) ** 2))
    brim = Mask(hbox).poly([H(140 * math.cos(t), 28 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 96)]).array()
    img *= 1.0 - 0.30 * blur(shift(brim, 4, 14), 7) * disc_alpha(hxh, hyh, hrh) * (1.0 - brim)
    over(img, brim, 0.10 + 0.10 * smoothstep(60.0, -120.0, U) * smoothstep(8.0, -20.0, V))
    cap_pts = [(-106, -156), (-84, -183), (-34, -197), (24, -195), (78, -186), (108, -158),
               (94, -143), (54, -138), (12, -146), (-30, -137), (-70, -146), (-98, -141)]
    cap = Mask(hbox).poly([H(u, v) for u, v in spline(cap_pts, closed=True, n=10)]).array()
    over(img, cap, lerp(0.99, 0.74, smoothstep(-200.0, -135.0, V)))

    # --- falling snow -----------------------------------------------------------------
    keep_out = blur(np.maximum(np.maximum(body, crown), np.maximum(brim, crescent)), 14)
    pts = scatter(rs, 95, 78.0, accept=lambda x, y: y < 1010 and keep_out[int(y), int(x)] < 0.03)
    dots = [(x, y, rs.uniform(6.0, 12.5), rs.uniform(0.70, 0.95)) for x, y in pts]
    over(img, soft_dots(dots), 1.0)

    vignette(img, 0.22)
    return img


# --------------------------------------------------------------------------------------
# 3. Reindeer against the moon
# --------------------------------------------------------------------------------------
def deer_masks(ox, oy, s):
    """Stag silhouette, facing right.

    Local coordinates: x forward (towards the head), y up, origin on the ground under
    the body; the shoulder is about 338 units high.  Returns (near, far) coverage masks -
    ``far`` holds the legs and antler on the far side, painted a touch lighter.
    """
    def T(x, y):
        return (ox + s * x, oy - s * y)

    bbox = (ox - 260 * s, oy - 860 * s, ox + 360 * s, oy + 8)

    def blob(mask, pts, n=12):
        mask.poly([T(x, y) for x, y in spline(pts, closed=True, n=n)])

    def limb(mask, joints):
        c = spline(np.asarray(joints, dtype=np.float64), n=10)
        mask.stroke([T(x, y) for x, y, _ in c], c[:, 2] * s)

    def hoof(mask, x):
        mask.poly([T(x - 14, 15), T(x + 10, 17), T(x + 19, 0), T(x - 16, 0)])

    def tine(mask, pts, r0, r1):
        c = spline(pts, n=14)
        mask.stroke([T(x, y) for x, y in c], (max(r0 * s, 4.8), max(r1 * s, 4.4)))

    def antler(mask, base, turn=0.0, k=1.0):
        """Main beam and tines; ``turn`` (radians) leans the whole antler forward."""
        ca, sa = math.cos(turn), math.sin(turn)

        def P(pts):
            return [(base[0] + k * (x * ca + y * sa), base[1] + k * (-x * sa + y * ca)) for x, y in pts]

        tine(mask, P([(0, 0), (-22, 46), (-52, 92), (-76, 146), (-82, 204), (-66, 256), (-36, 292)]),
             10.5 * k, 5.6)                                                # beam: back, up, then forward
        tine(mask, P([(-2, 12), (30, 20), (58, 44), (72, 82)]), 8.0 * k, 5.4)       # brow tine
        tine(mask, P([(-30, 60), (2, 72), (26, 100), (34, 130)]), 8.0 * k, 5.4)     # bez tine
        tine(mask, P([(-66, 122), (-32, 138), (-10, 170), (-4, 198)]), 8.0 * k, 5.4)  # trez tine
        tine(mask, P([(-82, 190), (-52, 208), (-38, 240)]), 7.5 * k, 5.4)           # royal tine
        tine(mask, P([(-74, 236), (-92, 262), (-98, 298)]), 7.5 * k, 5.4)           # back fork

    body_near, body_far, ant_near, ant_far = Mask(bbox), Mask(bbox), Mask(bbox), Mask(bbox)

    # far-side legs (a step ahead of the near ones)
    limb(body_far, [(146, 262, 29), (150, 220, 23), (158, 180, 16.5), (164, 135, 13.5), (166, 118, 14.5),
                    (168, 100, 11.3), (172, 62, 10), (174, 30, 11.8), (178, 15, 10.5)])
    hoof(body_far, 178)
    limb(body_far, [(-112, 285, 44), (-104, 250, 40), (-106, 222, 35), (-112, 190, 27), (-122, 158, 19.5),
                    (-132, 130, 15.5), (-131, 108, 12), (-126, 68, 10.5), (-121, 30, 12), (-117, 15, 10.5)])
    hoof(body_far, -117)
    antler(ant_far, (222, 504), turn=math.radians(22), k=0.93)

    # trunk: deep chest, tucked-up flank, rounded rump
    blob(body_near, [(92, 363), (40, 352), (-30, 345), (-100, 348), (-150, 354), (-188, 338), (-206, 300),
                     (-202, 258), (-188, 225), (-105, 238), (-40, 226), (30, 216), (100, 210), (140, 222),
                     (160, 255), (158, 300), (135, 350)])
    # neck (thick, with a mane) and head
    blob(body_near, [(40, 348), (100, 367), (130, 396), (152, 432), (170, 468), (184, 492), (205, 505),
                     (232, 499), (264, 480), (290, 462), (303, 448), (299, 434), (284, 428), (258, 428),
                     (238, 431), (226, 414), (216, 372), (200, 328), (176, 285), (140, 262), (80, 300)])
    # near-side legs
    limb(body_near, [(118, 262, 33), (116, 220, 26), (120, 180, 18), (122, 135, 14), (123, 118, 15),
                     (124, 100, 11.8), (125, 62, 10.5), (125, 30, 12.3), (129, 15, 11)])
    hoof(body_near, 129)
    limb(body_near, [(-155, 290, 52), (-148, 252, 46), (-150, 222, 40), (-156, 190, 30), (-166, 158, 21),
                     (-176, 130, 16), (-175, 108, 12.5), (-170, 68, 11), (-165, 30, 12.5), (-161, 15, 11)])
    hoof(body_near, -161)
    # tail and ear
    blob(body_near, [(-176, 346), (-204, 356), (-228, 352), (-206, 338), (-186, 326)], n=8)
    blob(body_near, [(200, 494), (180, 518), (152, 540), (162, 512), (182, 488)], n=8)
    antler(ant_near, (208, 506))

    def organic(mask, sigma=3.0):          # round the joins between the parts a little
        return smoothstep(0.38, 0.62, blur(mask.array(), sigma))

    near = np.maximum(organic(body_near), ant_near.array())
    far = np.maximum(organic(body_far), ant_far.array())
    return near, far


def ground_shadow(mask, ground_y, squash=0.5, shear=0.2):
    """Shadow of an upright shape cast on the ground towards the viewer."""
    im = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    coeffs = (1.0, -shear, shear * ground_y, 0.0, -1.0 / squash, ground_y * (1.0 + 1.0 / squash))
    out = im.transform((SIZE, SIZE), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def make_reindeer():
    rs = np.random.RandomState(20263)
    mx, my, R = 700.0, 640.0, 415.0
    crest = 996.0                                           # height of the hill top under the stag

    # --- night sky glowing around the moon ----------------------------------------------
    d = np.hypot(XX - mx, YY - my)
    out = np.maximum(d - R, 0.0)
    img = 0.13 + 0.30 * np.exp(-out / 105.0) + 0.10 * np.exp(-out / 360.0)

    # --- stars: soft dots and a few four-pointed sparkles --------------------------------
    sparkles = [(176, 236, 30), (1226, 190, 27), (1304, 604, 24), (90, 640, 24), (1052, 96, 23), (372, 104, 23),
                (700, 112, 20)]
    pts = scatter(rs, 46, 92.0, accept=lambda x, y: (math.hypot(x - mx, y - my) > R + 55 and y < crest - 40
                                                     and all(math.hypot(x - a, y - b) > 80 for a, b, _ in sparkles)))
    dots = [(x, y, rs.uniform(5.2, 9.0), rs.uniform(0.55, 1.0)) for x, y in pts]
    over(img, soft_dots(dots), 1.0)
    sp = Mask()
    for x, y, r in sparkles:
        sp.poly(star_points(x, y, r, r * 0.36, n=4))
    sp = sp.array()
    img += 0.25 * blur(sp, 9) * (1.0 - img)
    over(img, sp, 0.96)

    # --- the moon: bright disc, slightly darker limb and a few shallow craters --------------
    tone = 0.985 - 0.10 * (d / R) ** 3
    craters = np.zeros_like(img)
    for a, b, r in ((-0.45, -0.42, 0.17), (-0.10, -0.66, 0.09), (-0.68, 0.00, 0.10), (-0.27, -0.06, 0.07),
                    (0.70, -0.10, 0.10), (0.66, 0.30, 0.07)):
        px, py, pr = mx + a * R, my + b * R, r * R
        bowl = np.clip(0.5 - ((XX - px) + (YY - py)) / (2.8 * pr), 0.0, 1.0)      # deeper at upper left
        craters = np.maximum(craters, disc_alpha(px, py, pr) * (0.55 + 0.45 * bowl))
    tone = tone * (1.0 - 0.09 * blur(craters, 6))
    over(img, np.clip(R - d + 0.5, 0.0, 1.0), tone)

    # --- snowy hill crest -------------------------------------------------------------------
    def hill(x):
        return crest + 0.00016 * (x - 690.0) ** 2 + 5.0 * np.sin(x / 75.0 + 1.0)

    # firs on the skyline, either side of the moon (far ones lighter)
    paint_firs(img, [(60, 250, 140, 0.21), (292, 250, 136, 0.21), (1118, 230, 128, 0.21),
                     (1352, 270, 150, 0.21), (168, 340, 176, 0.125), (1236, 380, 192, 0.125)], hill, sink=24.0)

    # --- the stag ---------------------------------------------------------------------------
    ground_y, scale = crest + 2.0, 0.86
    near, far = deer_masks(652.0, ground_y, scale)
    over(img, far, 0.17)
    over(img, near, lerp(0.06, 0.11, smoothstep(ground_y - 640.0, ground_y, YY)))

    # --- hill in front, rim-lit by the moon, with the stag's shadow ---------------------------
    yh = hill(XX)
    ha = np.clip(YY - yh + 0.5, 0.0, 1.0)
    depth = YY - yh
    htone = lerp(0.52, 0.28, smoothstep(0.0, 400.0, depth))
    htone = htone + 0.30 * np.exp(-depth / 20.0) * (0.45 + 0.55 * np.exp(-((XX - mx) / 470.0) ** 2))
    htone = htone + 0.02 * smooth_noise(rs, 200)
    over(img, ha, htone)
    cast = ground_shadow(np.maximum(near, far), ground_y)
    cast = lerp(blur(cast, 4), blur(cast, 13), smoothstep(0.0, 190.0, depth))      # softer further away
    img *= 1.0 - 0.44 * cast * ha * (1.0 - smoothstep(60.0, 300.0, depth))
    # a nearer drift across the bottom, its crest catching a little light
    yd = 1292.0 + 34.0 * np.sin(XX / 310.0 + 0.5)
    img *= 1.0 - 0.10 * smoothstep(-80.0, 0.0, YY - yd) * (1.0 - np.clip(YY - yd + 0.5, 0.0, 1.0))
    img += 0.075 * np.clip(YY - yd + 0.5, 0.0, 1.0) * np.exp(-np.maximum(YY - yd, 0.0) / 70.0)

    vignette(img, 0.20)
    return img


# --------------------------------------------------------------------------------------
# 4. Greeting card
# --------------------------------------------------------------------------------------
def holly_leaf(bx, by, length, width, angle, spikes=4, bend=0.0, depth=0.5):
    """Holly leaf from its stem end (bx, by), pointing along ``angle``; ``bend`` arches
    the midrib upwards.  Returns (outline, the half facing the light, midrib)."""
    n = 141
    t = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * t) ** (0.6 + 0.6 * t)          # rounded at the stem, pointed at the tip
    scallop = 1.0 - depth * np.abs(np.sin(spikes * np.pi * t)) ** 0.85     # spikes where this is 1
    half = 0.5 * width * env * scallop
    along = t * length
    ca, sa = math.cos(angle), math.sin(angle)
    up = -1.0 if ca >= 0 else 1.0                       # sign of the across-leaf axis that points up
    bow = up * bend * length * np.sin(np.pi * t)

    def xy(u, v):
        return list(zip(bx + u * ca - v * sa, by + u * sa + v * ca))

    lit_sign = -1.0 if (-0.5 * sa + 0.85 * ca) > 0 else 1.0     # light comes from the upper left
    side_a, side_b, mid = xy(along, bow - half), xy(along, bow + half), xy(along, bow)
    lit = side_a if lit_sign < 0 else side_b
    return side_a + side_b[::-1], lit + mid[::-1], mid


def snowflake(mask, cx, cy, R, w, rot=0.0):
    """Chunky six-armed snowflake (arms w px wide) added to a Mask."""
    for k in range(6):
        a = rot + k * math.pi / 3.0
        ux, uy = math.cos(a), math.sin(a)
        mask.stroke([(cx, cy), (cx + ux * R, cy + uy * R)], w / 2.0)
        for frac, bl in ((0.44, 0.34), (0.72, 0.23)):
            px, py = cx + ux * R * frac, cy + uy * R * frac
            for sgn in (-1.0, 1.0):
                b = a + sgn * math.pi / 3.0
                mask.stroke([(px, py), (px + math.cos(b) * R * bl, py + math.sin(b) * R * bl)], w * 0.45)
    mask.poly(star_points(cx, cy, R * 0.20, R * 0.20 * 0.866, n=6, rot=rot))
    return mask


def make_greeting():
    rs = np.random.RandomState(20264)
    cx = 700.0

    # --- background: a soft pool of light that falls away to the edges --------------------
    d = np.hypot(XX - cx, (YY - 690.0) * 1.05)
    img = lerp(0.50, 0.14, smoothstep(40.0, 880.0, d))
    bokeh = scatter(rs, 16, 210.0, accept=lambda x, y: math.hypot(x - cx, y - 700) > 430)
    img += 0.05 * soft_dots([(x, y, rs.uniform(45, 90), 1.0) for x, y in bokeh], softness=0.5)

    # --- lettering -----------------------------------------------------------------------
    size = 200.0 * 1010.0 / max(text_width("Christmas", 200), 1.0)
    base1, base2, base3 = 640.0, 640.0 + 1.13 * size, 640.0 + 1.13 * size + 196.0
    merry, xmas = lettering([("Merry", cx, base1, size), ("Christmas", cx, base2, size)])
    (year,) = lettering([("2026", cx, base3, size * 0.80)])
    letters = np.maximum(np.maximum(merry, xmas), year)

    # --- holly sprig above the words --------------------------------------------------------
    hy = 346.0
    leaves = [holly_leaf(cx - 16, hy, 246, 128, math.radians(197), bend=0.08),
              holly_leaf(cx + 16, hy, 246, 128, math.radians(-17), bend=0.08),
              holly_leaf(cx, hy - 8, 178, 106, math.radians(-90), spikes=3)]
    hbox = (cx - 290, hy - 210, cx + 290, hy + 90)
    leaf_all, leaf_hi, rib = Mask(hbox), Mask(hbox), Mask(hbox)
    for outline, lit, mid in leaves:
        leaf_all.poly(outline)
        leaf_hi.poly(lit)
        rib.stroke(mid[10:-22], (4.6, 4.0))
    leaf_all, leaf_hi, rib = leaf_all.array(), leaf_hi.array(), rib.array()
    berries = [(cx, hy - 20, 32), (cx - 32, hy + 28, 32), (cx + 32, hy + 28, 32)]

    # --- snowflakes -------------------------------------------------------------------------
    flakes = Mask()
    flake_list = [(282, 404, 92, 12.0, 0.0), (1118, 404, 92, 12.0, math.pi / 6),
                  (250, 1104, 74, 11.0, math.pi / 6), (1150, 1104, 74, 11.0, 0.0),
                  (700, 1172, 54, 10.0, 0.0)]
    for x, y, R, w, rot in flake_list:
        snowflake(flakes, x, y, R, w, rot)
    flakes = flakes.array()
    swash = Mask()                                # a tapered rule and a little star either side of the year
    cols = np.nonzero(year.max(axis=0) > 0.5)[0]
    year_half = max(cx - cols.min(), cols.max() - cx) if len(cols) else 190.0
    for sgn in (-1.0, 1.0):
        xs = np.linspace(cx + sgn * (year_half + 34), cx + sgn * (year_half + 124), 24)
        swash.stroke(list(zip(xs, np.full(24, base3 - 40.0))), 4.2 + 2.6 * np.sin(np.linspace(0.0, math.pi, 24)))
        swash.poly(star_points(cx + sgn * (year_half + 156), base3 - 40.0, 21.0, 9.0, n=4))
    swash = swash.array()

    # --- composite: shadows first, then the bright things --------------------------------------
    everything = np.maximum.reduce([letters, leaf_all, flakes, swash])
    for bx_, by_, br_ in berries:
        everything = np.maximum(everything, disc_alpha(bx_, by_, br_))
    img *= 1.0 - 0.42 * blur(shift(everything, 5, 8), 11)

    over(img, flakes, 0.86)
    img += 0.10 * blur(flakes, 12) * (1.0 - img)
    over(img, swash, 0.84)

    over(img, leaf_all, 0.62)
    over(img, leaf_hi * leaf_all, 0.80)
    over(img, rib * leaf_all, 0.42)
    img *= 1.0 - 0.30 * blur(np.maximum.reduce([disc_alpha(x + 4, y + 7, r + 3) for x, y, r in berries]), 6) * leaf_all
    for bx_, by_, br_ in berries:
        ball(img, bx_, by_, br_, hi=1.0, lo=0.52)

    for mask, top, bottom in ((merry, base1 - 0.72 * size, base1 + 0.20 * size),
                              (xmas, base2 - 0.74 * size, base2),
                              (year, base3 - 0.60 * size, base3)):
        over(img, mask, lerp(1.0, 0.87, smoothstep(top, bottom, YY)))

    # --- a sprinkling of soft snow dots ----------------------------------------------------------
    keep_out = blur(everything, 22)
    pts = scatter(rs, 44, 112.0, accept=lambda x, y: keep_out[int(y), int(x)] < 0.02)
    dots = [(x, y, rs.uniform(5.5, 11.0), rs.uniform(0.55, 0.9)) for x, y in pts]
    over(img, soft_dots(dots), 0.95)
    return img


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
IMAGES = {
    "tree": (make_tree, "sample_tree.png"),
    "snowman": (make_snowman, "sample_snowman.png"),
    "reindeer": (make_reindeer, "sample_reindeer.png"),
    "greeting": (make_greeting, "sample_greeting.png"),
}


def finish(img):
    """Float picture (0..1) -> antialiased 8-bit greyscale Pillow image."""
    img = ndi.gaussian_filter(np.clip(img, 0.0, 1.0), 0.6, mode="nearest")
    return Image.fromarray(np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))


def main(argv=None):
    names = list(sys.argv[1:] if argv is None else argv) or list(IMAGES)
    unknown = [n for n in names if n not in IMAGES]
    if unknown:
        raise SystemExit(f"unknown image name(s): {', '.join(unknown)}; choose from {', '.join(IMAGES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.perf_counter()
        func, filename = IMAGES[name]
        im = finish(func())
        assert im.mode == "L" and im.size == (SIZE, SIZE)
        im.save(OUT_DIR / filename, optimize=True)
        a = np.asarray(im)
        p1, p50, p99 = np.percentile(a, [1, 50, 99])
        print(f"{filename:22s} p1={p1:5.1f}  median={p50:5.1f}  p99={p99:5.1f}  mean={a.mean():5.1f}  "
              f"min={a.min():3d}  max={a.max():3d}  ({time.perf_counter() - t0:.1f} s)")
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
