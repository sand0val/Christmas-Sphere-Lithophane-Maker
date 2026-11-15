"""Colours for the lit preview: how the ornament should look with a light inside.

Light passing through white PLA falls off roughly exponentially with the wall
thickness (Beer-Lambert), and gets warmer in colour the further it travels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Thickness of white PLA that cuts the transmitted light to 1/e (~37 %).
# Published figures for white PLA span roughly 1.2 mm (lithophane calculators) to
# 2.5 mm (spectrometer data for a very translucent "milky" PLA).  The lower end is
# used: it matches how lithophanes look in a dim room, where the eye adapts to
# the glow and the picture reads with more contrast than the raw numbers suggest.
ATTENUATION_MM = 1.2

LIGHT_COLORS = {
    "warm": (1.00, 0.88, 0.70),     # warm-white LED (~2700 K)
    "cool": (0.90, 0.95, 1.00),     # cool-white LED
}


def transmission(thickness: np.ndarray, t_ref: float) -> np.ndarray:
    """Linear luminance relative to a wall of thickness t_ref."""
    return np.exp(-(thickness - t_ref) / ATTENUATION_MM)


# Which face of the wall carries the relief matters.  White PLA scatters light:
# relief on the INSIDE (the lit face) shapes the light first, and the light then
# spreads sideways inside the wall before it leaves through a smooth outside, so
# the picture comes out blurred.  Relief on the OUTSIDE is the last thing the
# light meets and stays sharp.  The first ornament printed with the relief
# inside one half and outside the other showed exactly that; a Gaussian blur of
# about this radius, applied to the light that the inside relief lets through,
# reproduces both that print and a Monte Carlo simulation of light scattering
# in a 0.8 - 2.8 mm wall (1.5 mm dots keep ~60 % of their brightness, 1.2 mm
# dark details keep under half of their contrast).
INSIDE_BLUR_MM = 1.0


def soften_inside(in_shift: np.ndarray, row_mm: float, col_mm: np.ndarray) -> np.ndarray:
    """Relief on the inside of the wall as it effectively looks when lit.

    in_shift: how far the inside surface stands in from its plain sphere (mm;
    rows x columns of evenly spaced samples).  row_mm: spacing of the rows;
    col_mm: spacing of the columns in each row.  Returns the relief that,
    seen sharp, would let through the same light as the real one seen through
    the scattering wall."""
    light = np.exp(-np.asarray(in_shift, dtype=np.float64) / ATTENUATION_MM)
    light = ndimage.gaussian_filter1d(light, INSIDE_BLUR_MM / row_mm, axis=0, mode="nearest")
    for j, spacing in enumerate(np.broadcast_to(col_mm, (light.shape[0],))):
        light[j] = ndimage.gaussian_filter1d(light[j], INSIDE_BLUR_MM / max(float(spacing), 1e-3),
                                             mode="wrap")
    return -ATTENUATION_MM * np.log(np.maximum(light, 1e-12))


_LUT_SIZE = 1024
_LUT_SPAN_MM = 8.0          # thickness range covered by the lookup table, from t_min


def _glow_table(t_min: float, light: str) -> np.ndarray:
    """RGB for thicknesses t_min .. t_min + span, plus a last row for the inside."""
    t = t_min + np.linspace(0.0, _LUT_SPAN_MM, _LUT_SIZE)
    y = np.clip(transmission(t, t_min), 0.0, 1.0)
    base = np.array(LIGHT_COLORS.get(light, LIGHT_COLORS["warm"]))
    # thicker plastic absorbs blue first, so shadows drift towards amber
    exponents = np.array([1.00, 1.15, 1.45])
    lin = y[:, None] ** exponents[None, :] * base[None, :]
    rgb = np.clip(lin, 0.0, 1.0) ** (1 / 2.2)          # display gamma
    rgb = 0.03 + 0.97 * rgb                             # faint silhouette in the dark
    inside = np.clip(np.array([1.0, 0.97, 0.9]) * base ** 0.3, 0, 1)
    return (np.vstack([rgb, inside]) * 255).astype(np.uint8)


def glow_rgb(thickness: np.ndarray, t_min: float, light: str = "warm") -> np.ndarray:
    """uint8 RGB per vertex for the lit view.

    thickness: wall thickness per vertex in mm; NaN marks the inside of the
    shell (seen through the light opening), which is shown at full brightness.
    """
    table = _glow_table(t_min, light)
    inside = np.isnan(thickness)
    scaled = (np.where(inside, t_min, thickness) - t_min) * ((_LUT_SIZE - 1) / _LUT_SPAN_MM)
    index = np.clip(scaled, 0, _LUT_SIZE - 1).astype(np.intp)
    index[inside] = _LUT_SIZE
    return table[index]
