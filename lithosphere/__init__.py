"""Lithophane Christmas sphere generator.

Core (no GUI dependencies):
    params   - every user-facing parameter, limits, validation, derived geometry
    imaging  - photo loading, tone adjustments, projection onto the sphere
    overlay  - lettering and Christmas decorations laid over the pictures, and the
               decorations on the plain parts of the ball
    fonts    - the lettering fonts this computer has
    geometry - watertight printable mesh
    export   - STL / 3MF writers and the print-settings sheet
    checks   - printability analysis
    slicer   - optional Bambu Studio hand-off
"""

__version__ = "1.6.0"
APP_NAME = "Lithophane Sphere Studio"
