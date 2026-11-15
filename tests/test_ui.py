"""Tests that drive the real window - without ever putting it on the screen.

They need a working OpenGL context, so they only run when asked for:

    set LSPHERE_UI_TEST=1
    python -m pytest tests/test_ui.py -q
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(os.environ.get("LSPHERE_UI_TEST") != "1",
                                reason="set LSPHERE_UI_TEST=1 to run the window tests")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")


@pytest.fixture(scope="module")
def app():
    from lithosphere.ui import style
    from PyQt6 import QtWidgets

    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    style.apply(instance)
    return instance


@pytest.fixture(scope="module")
def window(app):
    """One window for the whole module, with every message box turned into a note."""
    from PyQt6 import QtCore, QtWidgets
    from lithosphere.ui import main_window as mw

    asked: list[str] = []
    box = QtWidgets.QMessageBox
    originals = (box.information, box.warning, box.question)
    box.information = staticmethod(lambda *a, **k: asked.append("info:" + str(a[2])))
    box.warning = staticmethod(lambda *a, **k: asked.append("warning:" + str(a[2])))
    box.question = staticmethod(lambda *a, **k: (asked.append("question:" + str(a[2])),
                                                 box.StandardButton.No)[1])
    w = mw.MainWindow(samples_dir=SAMPLES, session_path=None)
    w.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    w.show()
    w.asked = asked
    yield w
    box.information, box.warning, box.question = originals
    w.dirty = False
    w._worker = None
    w.close()
    app.processEvents()


@pytest.fixture()
def win(app, window):
    """The shared window, put back to a fresh demo ornament."""
    window._new_project(confirm=False)
    window._set_mode("lit")
    window._select_slot(0)
    window.asked.clear()
    settle(app, window)
    return window


def settle(app, window) -> None:
    window._timer.stop()
    window._sharp_timer.stop()
    window._rebuild()
    for _ in range(5):
        app.processEvents()


def wheel(app, widget) -> None:
    from PyQt6 import QtCore, QtGui

    centre = QtCore.QPointF(widget.width() / 2, widget.height() / 2)
    event = QtGui.QWheelEvent(centre, widget.mapToGlobal(centre), QtCore.QPoint(0, 0),
                              QtCore.QPoint(0, -120), QtCore.Qt.MouseButton.NoButton,
                              QtCore.Qt.KeyboardModifier.NoModifier,
                              QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    app.sendEvent(widget, event)
    app.processEvents()


def test_starts_clean_with_the_samples(app, win):
    assert win._error is None
    assert win.params.n_images == 3 and all(win.params.images[k].path for k in range(3))
    assert win.export_btn.isEnabled()
    assert "Ready to print" in win.verdict.text()


def test_relief_side_choices_say_what_they_cost(app, win):
    """The default puts the picture on the outside, which is sharp when lit.  The other two
    choices say on the checklist that they look soft, and switching print-safe edges off
    where relief can hang is flagged on the checklist and in the note about supports."""
    def titles():
        return [title for _level, title, _detail in win._checks_shown]

    assert win.params.relief_side == "outside" and win.relief.current() == "outside"
    assert "Sharp picture, nothing overhangs" in titles() and "Ready to print" in win.verdict.text()
    assert "floating" not in win.supports_note.text()
    win._set_param("relief_side", "split")
    settle(app, win)
    assert "Upper half of the pictures will look soft" in titles() and "to check" in win.verdict.text()
    win._set_param("relief_side", "inside")
    settle(app, win)
    assert "The pictures will look soft" in titles() and "to check" in win.verdict.text()
    win._set_param("safe_edges", False)
    settle(app, win)
    assert "Details may print in mid-air" in titles() and "floating" in win.supports_note.text()
    win._set_param("relief_side", "split")              # on both sides nothing can hang even then
    settle(app, win)
    assert "Details may print in mid-air" not in titles() and "floating" not in win.supports_note.text()
    win._set_param("relief_side", "outside")            # outside it can, until the edges are safe again
    settle(app, win)
    assert "Details may print in mid-air" in titles() and "floating" in win.supports_note.text()
    win._set_param("safe_edges", True)
    settle(app, win)
    assert "Ready to print" in win.verdict.text() and "floating" not in win.supports_note.text()


def test_sessions_with_an_old_default_relief_move_to_outside(app, win, tmp_path):
    import json

    from lithosphere import __version__
    from lithosphere.params import Params

    last_dir = win._last_dir
    path = tmp_path / "session.json"
    try:
        # the first version ("inside") and versions 1.1 / 1.2 ("split", as it was printed)
        for header, side in (({}, "inside"), ({"version": "1.2.0"}, "split")):
            old = Params(relief_side=side, diameter=90.0).to_dict()
            path.write_text(json.dumps({**header, "params": old, "dirty": False}), encoding="utf-8")
            assert win._load_file(str(path), as_session=True)
            assert win.params.relief_side == "outside" and win.params.diameter == 90.0
            assert win.relief.current() == "outside"
        win._set_param("relief_side", "split")          # a choice made now is kept from then on
        win.session_path = str(path)
        win._save_session()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["version"] == __version__ and saved["params"]["relief_side"] == "split"
        assert win._load_file(str(path), as_session=True) and win.params.relief_side == "split"
    finally:
        win.session_path = None
        win._last_dir = last_dir


def test_gap_default_and_what_the_checklist_says_about_the_seam(app, win, tmp_path):
    """New ornaments get a 12 degree gap, and so does a session that still carries the
    old default of 6.  The checklist says where the print seam goes for the gap chosen."""
    import json

    from lithosphere import __version__
    from lithosphere.params import Params

    def titles():
        return [title for _level, title, _detail in win._checks_shown]

    assert win.params.gap_deg == 12.0 and win.sliders["gap_deg"].value() == 12.0
    assert "Seam hidden" in titles() and "Ready to print" in win.verdict.text()
    win._set_param("gap_deg", 0.0)
    settle(app, win)
    assert "Seam between two pictures" in titles() and "to check" in win.verdict.text()
    win._set_param("gap_deg", 1.5)
    settle(app, win)
    assert "Seam touches the pictures" in titles()
    win._reset_params(("gap_deg",))
    settle(app, win)
    assert win.params.gap_deg == 12.0 and "Seam hidden" in titles()

    old = Params(gap_deg=6.0, relief_side="outside").to_dict()
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"version": "1.1.0", "params": old, "dirty": False}),
                    encoding="utf-8")
    last_dir = win._last_dir
    try:
        assert win._load_file(str(path), as_session=True)
        assert win.params.gap_deg == 12.0 and win.sliders["gap_deg"].value() == 12.0
        assert win.params.relief_side == "outside"          # nothing else is touched
        win._set_param("gap_deg", 6.0)                      # a gap chosen now is kept from then on
        win.session_path = str(path)
        win._save_session()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["version"] == __version__ and saved["params"]["gap_deg"] == 6.0
        assert win._load_file(str(path), as_session=True) and win.params.gap_deg == 6.0
    finally:
        win.session_path = None
        win._last_dir = last_dir


def test_mouse_wheel_never_edits_a_control(app, win):
    before = win.params.to_dict()
    win.tabs.setCurrentIndex(0)
    for slider in list(win.slot_sliders.values()) + list(win.sliders.values()):
        wheel(app, slider.slider)
        wheel(app, slider.spin)
    for combo in (win.light_combo_preset, win.hanger_combo, win.quality_combo, win.light_combo):
        wheel(app, combo)
    assert win.params.to_dict() == before and not win.dirty


def test_cap_keeps_its_proportion_however_the_ball_is_resized(app, win):
    start = (win.params.cap_diameter, win.params.cap_height)
    win._set_param("diameter", 150.0)
    assert win.params.cap_diameter == pytest.approx(150 * 26 / 80, abs=0.25)
    for d in list(range(149, 79, -1)):                 # a slow drag back, one millimetre at a time
        win._set_param("diameter", float(d))
    assert (win.params.cap_diameter, win.params.cap_height) == start
    win._set_param("diameter", 150.0)
    win._reset_params(("diameter", "band_height_deg", "gap_deg", "projection", "relief_side"))
    assert (win.params.diameter, win.params.cap_diameter, win.params.cap_height) == (80.0, *start)
    # a cap the user chose keeps the proportion they gave it
    win._set_param("cap_diameter", 20.0)
    win._set_param("diameter", 120.0)
    assert win.params.cap_diameter == pytest.approx(30.0, abs=0.25)


@pytest.mark.parametrize("edits", [
    [("t_frame", 4.0), ("hole_diameter", "max")],
    [("t_frame", 4.0), ("diameter", 70.0), ("preset", "c9")],
    [("cap_wall", 4.0), ("cap_height", 4.0)],
    [("diameter", 50.0), ("preset", "c9"), ("flare_angle", 25.0), ("t_frame", 4.0)],
    [("relief_side", "outside"), ("t_max", 4.0), ("t_frame", 0.8), ("band_height_deg", "max")],
    [("preset", "puck"), ("diameter", 50.0), ("flare_angle", 25.0), ("t_frame", 4.0)],
    [("preset", "puck"), ("seat_depth", "max"), ("cap_height", 4.0), ("cap_wall", 4.0)],
    [("seat", True), ("seat_diameter", "max"), ("hole_diameter", "max"), ("diameter", 150.0)],
    [("diameter", 150.0), ("seat", True), ("seat_diameter", "max"), ("flare_angle", 50.0),
     ("seat_diameter", "max"), ("diameter", 50.0)],
    [("preset", "puck"), ("preset", "c9"), ("preset", "puck"), ("preset", "mini")],
])
def test_linked_settings_never_reach_the_error_banner(app, win, edits):
    from lithosphere.params import derive

    for name, value in edits:
        if name == "preset":
            win.light_combo_preset.setCurrentIndex(win.light_combo_preset.findData(value))
            win._preset_chosen(0)
        else:
            if value == "max":
                value = win.sliders[name].spin.maximum()
            win._set_param(name, value)
        settle(app, win)
        assert win._error is None, (name, value, win._error)
        derive(win.params)
        for key, slider in win.sliders.items():       # every slider still shows its real value
            assert slider.value() == pytest.approx(getattr(win.params, key), abs=1e-6), key


def test_rotate_buttons_follow_the_eye_after_mirror(app, win):
    from lithosphere import imaging

    marker = np.zeros((4, 6), dtype=np.float32)
    marker[0, 0] = 1.0                                  # bright top-left corner
    slot = win._slot()

    def corner() -> tuple[bool, bool]:
        shown = imaging.oriented(marker, slot)
        y, x = np.argwhere(shown == 1.0)[0]
        return bool(y < shown.shape[0] / 2), bool(x < shown.shape[1] / 2)     # (top?, left?)

    assert corner() == (True, True)
    win._rotate(90)                                     # clockwise: top-left goes to top-right
    assert corner() == (True, False)
    win._rotate(-90)
    win._mirror()                                       # now it is top-right ...
    assert corner() == (True, False)
    win._rotate(90)                                     # ... and clockwise takes it bottom-right
    assert corner() == (False, False)
    win._rotate(-90)
    assert corner() == (True, False)


def test_cutaway_stays_face_on_when_a_picture_is_chosen(app, win):
    win._set_mode("cutaway")
    for index in (0, 1, 2, 1):
        win._select_slot(index)
        app.processEvents()
        cam = win.preview.plotter.camera
        view = np.array(cam.focal_point) - np.array(cam.position)
        view /= np.linalg.norm(view)
        normal = np.array(win.preview._cut_plane.GetNormal())
        assert abs(float(view @ normal)) > 0.95, index     # looking into the cut, not along it
        centre = win._preview_data.layout.centers[index]
        assert abs(math.remainder(win.preview._cut_longitude() - centre, 2 * math.pi)) < 0.01
    win._set_mode("cutaway")                               # clicking it again only re-frames
    assert win.preview.mode == "cutaway"
    win._set_mode("lit")


def test_preview_keeps_up_during_a_fast_drag(app, win, monkeypatch):
    calls = []
    real = win._rebuild
    monkeypatch.setattr(win, "_rebuild", lambda sharp=False: (calls.append(sharp), real(sharp))[1])
    win._timer.timeout.disconnect()
    win._timer.timeout.connect(lambda: win._rebuild())
    end = time.time() + 1.2
    value = 1.0
    while time.time() < end:                               # a change every few milliseconds
        value = 1.0 + (value - 0.99) % 0.5
        win._set_slot_param("zoom", value)
        app.processEvents()
        time.sleep(0.004)
    assert calls.count(False) >= 4, calls                  # it redraws while the control is moving


def test_dropping_more_photos_than_fit_wraps_round_and_says_so(app, win):
    files = sorted(os.path.join(SAMPLES, f) for f in os.listdir(SAMPLES) if f.endswith(".png"))
    win._drop_photos(2, [files[3], files[0], files[1]])    # three photos onto the third card
    paths = [os.path.basename(s.path) for s in win.params.images]
    assert paths[2] == os.path.basename(files[3]) and paths[3] == os.path.basename(files[0])
    assert paths[0] == os.path.basename(files[1])          # carried on round to picture 1
    assert win.params.n_images == 4
    win.asked.clear()
    win._drop_photos(0, files + files[:2])                 # six photos: two cannot be used
    assert any("were not" in a for a in win.asked)


def test_export_asks_before_printing_a_placeholder(app, win, monkeypatch):
    from PyQt6 import QtWidgets

    win.params.images[1].path = os.path.join(SAMPLES, "moved_away.png")
    opened = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (opened.append(1), ("", ""))[1]))
    win.asked.clear()
    win._export()
    assert any(a.startswith("question:") and "moved_away.png" in a for a in win.asked)
    assert not opened and win._worker is None              # answered No: nothing was started


def test_export_box_says_where_the_seam_ended_up(app, win, monkeypatch, tmp_path):
    """After slicing, the box reports where the layers really start - plainly when the
    seam is on the divider, as a warning when part of it landed in a picture."""
    from PyQt6 import QtWidgets

    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: shown.append((self.icon(), self.informativeText())) or 0)
    seam = {"loops": 1000, "divider_deg": 12.0, "stray": 0, "on_divider": 1.0, "worst_deg": 1.2,
            "worst_mm": 0.7}
    info = {"path": str(tmp_path / "x.stl"), "sheet": str(tmp_path / "x_print_settings.txt"),
            "stats": {"size_mm": (80.0, 80.0, 87.7), "weight_g": 50.0, "triangles": 1000},
            "bytes": 1, "project": str(tmp_path / "x_bambu_A1.3mf"), "problems": [], "tips": 0,
            "slice": {"ok": True, "print_time_s": 17000, "filament_g": 49.7, "layer_count": 730,
                      "support_used": False, "warnings": [], "seam": seam}}
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Information
    assert "down the middle of the divider at the back" in text and "0.7 mm" in text

    seam.update(stray=500, on_divider=0.5, worst_mm=7.7)
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Warning
    assert "not fully hidden" in text and "500 of the 1000 wall loops" in text

    info["slice"]["seam"] = None                # the check could not be made: say nothing
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Information and "seam" not in text.lower()

    # a few loops that Bambu Studio starts at a small island of the relief are reported for
    # what they are - not as a seam that a wider gap would hide, and not as a fault
    info["slice"]["seam"] = dict(seam, stray=2, elsewhere=2, on_divider=0.998, worst_deg=95.0,
                                 worst_mm=42.0, layers=500, broken=0)
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Information
    assert "except for 2 of the 1000 wall loops" in text and "wider gap" not in text

    # layers that are not a closed ring of two loops are a warning, with what to do
    info["slice"]["seam"] = dict(seam, stray=0, elsewhere=0, on_divider=1.0, worst_mm=0.7,
                                 layers=500, broken=5, loops=990)
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Warning
    assert "In 5 of the 500 layers" in text and "thinnest wall" in text
    assert "down the middle of the divider at the back" in text

    # a project made for the fine nozzle says so before anything else can go wrong
    info["slice"]["seam"] = dict(seam, stray=0, elsewhere=0, on_divider=1.0, worst_mm=0.7,
                                 layers=500, broken=0)
    win._export_done(info)
    assert "0.2 mm nozzle" not in shown[-1][1]
    win._set_param("nozzle", "0.2")
    win._export_done(info)
    icon, text = shown[-1]
    assert icon == QtWidgets.QMessageBox.Icon.Information
    assert "for the 0.2 mm nozzle" in text and "printer's screen" in text


def test_busy_dialog_cannot_be_dismissed_until_finished(app, win):
    from lithosphere.ui.main_window import _BusyDialog

    box = _BusyDialog("working", "", 0, 0, win)
    box.setCancelButton(None)
    box.setAttribute(__import__("PyQt6").QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    box.show()
    box.reject()
    box.close()
    assert box.isVisible()
    box.finish()
    assert not box.isVisible()


def test_stray_keys_do_not_change_the_3d_view(app, win):
    from PyQt6 import QtCore, QtGui

    pl = win.preview.plotter
    actor = win.preview._actor
    before = actor.GetProperty().GetRepresentation()
    position = tuple(pl.camera.position)
    for key in "w3vsrf":                                   # VTK's own single-key shortcuts
        code = getattr(QtCore.Qt.Key, "Key_" + key.upper())
        for kind in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
            app.sendEvent(win.preview.canvas, QtGui.QKeyEvent(
                kind, code, QtCore.Qt.KeyboardModifier.NoModifier, key))
    app.processEvents()
    assert actor.GetProperty().GetRepresentation() == before
    assert not pl.ren_win.GetStereoRender()
    assert tuple(pl.camera.position) == position


def test_window_never_presents_opengl_to_the_screen(app, win):
    """The 3D view is an ordinary picture: no OpenGL surface for an overlay to hook."""
    from PyQt6 import QtGui, QtWidgets

    assert win.windowHandle().surfaceType() == QtGui.QSurface.SurfaceType.RasterSurface
    assert not [w for w in win.findChildren(QtWidgets.QWidget) if w.inherits("QOpenGLWidget")]
    render_window = win.preview.plotter.render_window
    assert render_window.GetOffScreenRendering() and not render_window.GetShowWindow()
    assert render_window.GetUseOffScreenBuffers()          # so VTK never calls SwapBuffers either


def camera(win):
    """(focal point, distance, longitude, elevation) of the preview camera."""
    cam = win.preview.plotter.camera
    focal = np.array(cam.focal_point, dtype=float)
    offset = np.array(cam.position, dtype=float) - focal
    dist = float(np.linalg.norm(offset))
    return focal, dist, math.atan2(offset[1], offset[0]), math.asin(offset[2] / dist)


def drag(app, widget, start, end, button=None, shift=False, steps=5) -> None:
    """A mouse drag delivered straight to the widget (the real cursor never moves)."""
    from PyQt6 import QtCore, QtGui

    button = button or QtCore.Qt.MouseButton.LeftButton
    none = QtCore.Qt.MouseButton.NoButton
    mods = QtCore.Qt.KeyboardModifier.ShiftModifier if shift else QtCore.Qt.KeyboardModifier.NoModifier

    def send(kind, pos, btn, buttons):
        point = QtCore.QPointF(*pos)
        app.sendEvent(widget, QtGui.QMouseEvent(kind, point, widget.mapToGlobal(point), btn, buttons, mods))

    send(QtCore.QEvent.Type.MouseButtonPress, start, button, button)
    for k in range(1, steps + 1):
        f = k / steps
        send(QtCore.QEvent.Type.MouseMove,
             (start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f), none, button)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, button, none)
    app.processEvents()


def scroll(app, widget, notches: float) -> None:
    from PyQt6 import QtCore, QtGui

    centre = QtCore.QPointF(widget.width() / 2, widget.height() / 2)
    app.sendEvent(widget, QtGui.QWheelEvent(
        centre, widget.mapToGlobal(centre), QtCore.QPoint(0, 0), QtCore.QPoint(0, round(120 * notches)),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase, False))
    app.processEvents()


def picture_bytes(canvas) -> bytes:
    image = canvas.image()
    return bytes(image.constBits().asarray(image.sizeInBytes()))


def test_picture_fills_the_view_and_follows_its_size(app, win):
    canvas = win.preview.canvas
    image = canvas.image()
    assert image is not None and (image.width(), image.height()) == canvas.pixel_size()
    assert image.devicePixelRatioF() == pytest.approx(canvas.devicePixelRatioF())
    # not a blank frame: the glowing ornament is far brighter than the dark background
    grey = image.convertToFormat(image.Format.Format_Grayscale8)
    middle = grey.pixelColor(grey.width() // 2, grey.height() // 2).value()
    corner = grey.pixelColor(3, 3).value()
    assert middle > corner + 40
    size = win.size()
    try:
        win.resize(size.width() - 140, size.height() - 90)
        for _ in range(5):
            app.processEvents()
        image = canvas.image()
        assert (image.width(), image.height()) == canvas.pixel_size()
    finally:
        win.resize(size)
        for _ in range(5):
            app.processEvents()


def test_dragging_turns_the_ornament_the_way_the_mouse_moves(app, win):
    canvas = win.preview.canvas
    w, h = canvas.width(), canvas.height()
    focal, dist, lon, el = camera(win)
    before = picture_bytes(canvas)
    drag(app, canvas, (w * 0.5, h * 0.5), (w * 0.75, h * 0.5))          # a quarter of the width
    focal2, dist2, lon2, el2 = camera(win)
    assert math.remainder(lon2 - (lon - math.pi / 4), 2 * math.pi) == pytest.approx(0.0, abs=1e-6)
    assert el2 == pytest.approx(el, abs=1e-9) and dist2 == pytest.approx(dist, rel=1e-9)
    assert np.allclose(focal2, focal)
    assert tuple(win.preview.plotter.camera.up) == (0.0, 0.0, 1.0)
    drag(app, canvas, (w * 0.5, h * 0.2), (w * 0.5, h * 0.2 + h / 6))   # down: look from higher up
    assert camera(win)[3] == pytest.approx(el + math.pi / 6, abs=1e-6)
    drag(app, canvas, (w * 0.5, 0.0), (w * 0.5, h * 3.0))               # far too far: stops short of the pole
    _, dist3, _, el3 = camera(win)
    assert el3 == pytest.approx(math.radians(88.0), abs=1e-6) and dist3 == pytest.approx(dist, rel=1e-9)
    assert win.preview.plotter.camera.position[2] > 0
    after = canvas.image()
    assert (after.width(), after.height()) == canvas.pixel_size()
    assert picture_bytes(canvas) != before                              # and the picture was redrawn


def test_scrolling_zooms_within_sensible_limits(app, win):
    from PyQt6 import QtCore

    canvas = win.preview.canvas
    size = win.preview._size()
    _, dist, lon, el = camera(win)
    assert dist == pytest.approx(2.9 * size, rel=1e-6)
    scroll(app, canvas, 3)                                              # towards the ornament
    _, closer, lon2, el2 = camera(win)
    assert closer == pytest.approx(dist / 1.1 ** 3, rel=1e-6)
    assert (lon2, el2) == (pytest.approx(lon, abs=1e-9), pytest.approx(el, abs=1e-9))
    scroll(app, canvas, 200)
    assert camera(win)[1] == pytest.approx(0.62 * size, rel=1e-6)       # never inside the ornament
    scroll(app, canvas, -400)
    assert camera(win)[1] == pytest.approx(12.0 * size, rel=1e-6)       # never lost in the distance
    drag(app, canvas, (50.0, 50.0), (50.0, 10.0), button=QtCore.Qt.MouseButton.RightButton)
    assert camera(win)[1] == pytest.approx(12.0 * size / 1.01 ** 40, rel=1e-6)   # right-drag up = closer


def test_shift_drag_slides_the_view_and_look_at_brings_it_back(app, win):
    canvas = win.preview.canvas
    h = canvas.height()
    focal, dist, lon, el = camera(win)
    drag(app, canvas, (200.0, 200.0), (260.0, 170.0), shift=True)
    focal2, dist2, lon2, el2 = camera(win)
    assert dist2 == pytest.approx(dist, rel=1e-9)
    assert (lon2, el2) == (pytest.approx(lon, abs=1e-9), pytest.approx(el, abs=1e-9))
    per_pixel = 2.0 * dist * math.tan(math.radians(26.0) / 2.0) / h
    assert np.linalg.norm(focal2 - focal) == pytest.approx(math.hypot(60, 30) * per_pixel, rel=1e-6)
    view = -np.array([math.cos(el) * math.cos(lon), math.cos(el) * math.sin(lon), math.sin(el)])
    assert abs(float((focal2 - focal) @ view)) < 1e-6                   # sideways only, never in or out
    right = np.cross(view, (0.0, 0.0, 1.0))
    assert float((focal2 - focal) @ right) < 0                          # camera goes left, ornament right
    scroll(app, canvas, 4)
    win.preview.look_at_picture(0)                                      # a "Look at" button
    focal3, dist3, _, _ = camera(win)
    assert np.allclose(focal3, focal) and dist3 == pytest.approx(dist, rel=1e-6)


def test_spin_turns_the_view_until_switched_off(app, win):
    _, dist, lon, el = camera(win)
    win.spin_btn.setChecked(True)
    end = time.time() + 0.4
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)
    win.spin_btn.setChecked(False)
    _, dist2, lon2, el2 = camera(win)
    assert abs(math.remainder(lon2 - lon, 2 * math.pi)) > math.radians(1.5)
    assert dist2 == pytest.approx(dist, rel=1e-6) and el2 == pytest.approx(el, abs=1e-6)
    app.processEvents()
    assert camera(win)[2] == pytest.approx(lon2, abs=1e-9)              # and it really has stopped


def test_mesh_is_rebuilt_when_the_grid_layout_changes(app, win):
    assert len(win.preview._signature) == 6
    body = win._preview_data.body
    assert win.preview._signature[3:] == (body.n_outer, body.outer_rows, body.inner_rows)
    assert win.preview._mesh.n_cells == len(body.faces)


def test_no_page_is_wider_than_the_side_panel(app, win):
    """The side panel has a fixed width and does not scroll sideways, so one label that is
    too long pushes the right-hand edge of every control on its page out of sight."""
    from PyQt6 import QtWidgets

    checked = 0
    for index in range(win.tabs.count()):
        win.tabs.setCurrentIndex(index)
        settle(app, win)
        area = win.tabs.widget(index)
        assert isinstance(area, QtWidgets.QScrollArea)
        need, room = area.widget().minimumSizeHint().width(), area.viewport().width()
        assert need <= room, f"page {index + 1} needs {need} px but the panel shows {room} px"
        checked += 1
    assert checked == 6
    # ... and the six steps fit the tab bar without scroll arrows
    labels = [win.tabs.tabText(i).replace("&&", "&") for i in range(win.tabs.count())]
    assert labels == ["1 Pictures", "2 Decorate", "3 Sphere", "4 Light & dark", "5 Cap", "6 Print"]
    assert win.tabs.tabBar().sizeHint().width() <= win.tabs.width()
    win.tabs.setCurrentIndex(0)
    win.verdict.click()                                 # "Ready to print" opens the checklist
    assert win.tabs.currentIndex() == 5
    win.tabs.setCurrentIndex(0)


def _choose_light(win, key: str) -> None:
    win.light_combo_preset.setCurrentIndex(win.light_combo_preset.findData(key))
    win._preset_chosen(0)


def test_puck_preset_makes_a_seat_and_the_cap_follows(app, win):
    from lithosphere.params import LIGHT_PRESETS, cap_diameter_choices, derive

    def titles():
        return [title for _level, title, _detail in win._checks_shown]

    puck = LIGHT_PRESETS["puck"]
    p = win.params
    assert not p.seat and not win.seat_check.isChecked()
    assert not win.sliders["seat_diameter"].isEnabled() and not win.sliders["seat_depth"].isEnabled()
    assert "Seat for the light" not in titles()

    _choose_light(win, "puck")
    settle(app, win)
    p = win.params
    assert win._error is None and not win.asked
    assert p.seat and p.light_preset == "puck" and win.seat_check.isChecked()
    assert (p.seat_diameter, p.seat_depth, p.hole_diameter) == (puck.seat, puck.depth, puck.hole)
    assert win.sliders["seat_diameter"].isEnabled() and win.sliders["seat_depth"].isEnabled()
    assert p.cap_diameter == pytest.approx(puck.seat + 5.0)             # a 2.5 mm rim round it
    assert p.cap_height >= puck.depth + 2.5                              # the cap grew for the pocket
    assert "Seat for the light" in titles() and "Ready to print" in win.verdict.text()
    lay = derive(p)
    assert lay.seat and lay.r_seat == pytest.approx(puck.seat / 2)
    assert win.light_combo_preset.currentData() == "puck"
    # the guide tells how to use it
    sheet = " ".join(win._preview_data and __import__("lithosphere").export.settings_sheet(p, lay).split())
    assert "into the seat on top of the cap" in sheet

    # the opening can never take the ledge away from under the light
    assert win.sliders["hole_diameter"].spin.maximum() == pytest.approx(puck.seat - 3.0, abs=0.26)
    win._set_param("hole_diameter", win.sliders["hole_diameter"].spin.maximum())
    settle(app, win)
    assert win._error is None and p.seat and p.light_preset == "custom"
    assert win.light_combo_preset.currentData() == "custom"
    _choose_light(win, "puck")

    # a wider light takes its ledge and its rim along
    win._set_param("seat_diameter", puck.seat + 1.0)
    settle(app, win)
    assert win._error is None and p.light_preset == "custom"
    assert p.hole_diameter == pytest.approx(puck.hole + 1.0) and p.cap_diameter == pytest.approx(puck.seat + 6.0)
    lo, hi = cap_diameter_choices(p)
    assert win.sliders["cap_diameter"].spin.minimum() == pytest.approx(lo)
    assert lo == pytest.approx(p.seat_diameter + 2.4, abs=0.11)          # never less than a wall round it

    # switching the seat off leaves an ordinary opening, and on again a ledge
    win.seat_check.click()
    settle(app, win)
    assert win._error is None and not p.seat and not win.sliders["seat_depth"].isEnabled()
    assert "Seat for the light" not in titles()
    assert win.sliders["hole_diameter"].spin.maximum() > puck.seat
    win._set_param("hole_diameter", 31.0)
    win.seat_check.click()
    settle(app, win)
    assert win._error is None and p.seat and p.hole_diameter == pytest.approx(p.seat_diameter - 5.0, abs=0.26)

    # an ordinary bulb preset takes the seat away again; Reset puts the whole page back
    _choose_light(win, "balloon")
    settle(app, win)
    assert not p.seat and p.hole_diameter == 11.6 and p.light_preset == "balloon"
    _choose_light(win, "puck")
    win._reset_params(("light_preset", "hole_diameter", "seat", "seat_diameter", "seat_depth",
                       "cap_diameter", "cap_height", "cap_wall", "flare_angle", "hanger", "hanger_hole"))
    settle(app, win)
    from lithosphere.params import Params
    fresh = Params()
    for name in ("light_preset", "hole_diameter", "seat", "seat_diameter", "seat_depth", "cap_diameter",
                 "cap_height"):
        assert getattr(p, name) == getattr(fresh, name), name


def test_a_seat_that_needs_room_gets_a_steeper_shoulder_and_says_so(app, win):
    from lithosphere.params import LIGHT_PRESETS, cap_diameter_range

    def titles():
        return [title for _level, title, _detail in win._checks_shown]

    puck, tea = LIGHT_PRESETS["puck"], LIGHT_PRESETS["tealight"]
    p = win.params
    # a tea light is wider than the standard ball's cap can be with its 30 degree shoulder
    assert cap_diameter_range(p)[1] < tea.seat + 2.4 and p.flare_angle == 30.0
    _choose_light(win, "tealight")
    settle(app, win)
    assert win._error is None and p.seat and p.light_preset == "tealight"
    assert (p.seat_diameter, p.seat_depth, p.hole_diameter) == (tea.seat, tea.depth, tea.hole)
    assert 33.0 < p.flare_angle <= 37.0 and p.cap_diameter >= tea.seat + 2.4
    assert len(win.asked) == 1 and win.asked[0].startswith("info:A 39 mm seat needs a wider cap")
    assert f"raised to {p.flare_angle:.0f}" in win.asked[0] and "30" in win.asked[0]
    assert win.sliders["flare_angle"].value() == p.flare_angle
    assert "Seat for the light" in titles() and "Ready to print" in win.verdict.text()
    # choosing it again changes nothing and says nothing
    win.asked.clear()
    angle = p.flare_angle
    _choose_light(win, "tealight")
    settle(app, win)
    assert not win.asked and p.flare_angle == angle

    # the puck on a smaller ball: same remedy
    win._reset_params(("light_preset", "hole_diameter", "seat", "seat_diameter", "seat_depth",
                       "cap_diameter", "cap_height", "cap_wall", "flare_angle", "hanger", "hanger_hole"))
    win._set_param("diameter", 60.0)
    settle(app, win)
    win.asked.clear()
    _choose_light(win, "puck")
    settle(app, win)
    assert win._error is None and p.seat and p.seat_diameter == puck.seat and p.light_preset == "puck"
    assert p.flare_angle > 30.0 and len(win.asked) == 1 and "60 mm sphere" in win.asked[0]
    assert "raised to" in win.asked[0]

    # a ball too small for the light at any shoulder angle: the seat shrinks, and the box says why
    win._set_param("diameter", 50.0)
    win._set_param("flare_angle", 30.0)
    settle(app, win)
    win.asked.clear()
    _choose_light(win, "tealight")
    settle(app, win)
    assert win._error is None
    assert len(win.asked) == 1 and win.asked[0].startswith("info:A 39 mm seat does not fit")
    assert "50 mm sphere" in win.asked[0] and "sphere bigger" in win.asked[0]
    assert (not p.seat) or (p.seat_diameter < tea.seat and p.light_preset == "custom"
                            and p.hole_diameter <= p.seat_diameter - 3.0)

    # on the standard ball the puck fits as it is: nothing to say
    win._new_project(confirm=False)
    settle(app, win)
    p = win.params
    win.asked.clear()
    _choose_light(win, "puck")
    settle(app, win)
    assert not win.asked and p.seat_diameter == puck.seat and p.light_preset == "puck"
    assert p.flare_angle == 30.0
    # making the ball smaller afterwards shrinks the seat with it, without an error
    for d in range(79, 49, -1):
        win._set_param("diameter", float(d))
    settle(app, win)
    assert win._error is None and p.diameter == 50.0
    assert (not p.seat) or p.seat_diameter <= p.cap_diameter - 2.4 + 1e-6


def test_seat_survives_a_session_and_shows_in_every_view(app, win, tmp_path):
    import json

    _choose_light(win, "puck")
    settle(app, win)
    before = win.params.to_dict()
    path = tmp_path / "session.json"
    last_dir = win._last_dir
    try:
        win.session_path = str(path)
        win._save_session()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["params"]["seat"] is True and saved["params"]["light_preset"] == "puck"
        win._new_project(confirm=False)
        assert not win.params.seat
        assert win._load_file(str(path), as_session=True)
        assert win.params.to_dict() == before and win.seat_check.isChecked()
    finally:
        win.session_path = None
        win._last_dir = last_dir
    for mode in ("lit", "unlit", "cutaway", "bed"):
        win._set_mode(mode)
        settle(app, win)
        image = win.preview.canvas.image()
        assert image is not None and image.width() > 100
    win._set_mode("lit")


def test_nozzle_choice_reaches_the_checklist_the_settings_and_the_session(app, win, tmp_path):
    import json

    def shown():
        return {name: value for name, value, _why in win._settings_shown}

    def details():
        return " ".join(detail for _level, _title, detail in win._checks_shown)

    assert win.params.nozzle == "0.4" and win.nozzle_seg.current() == "0.4"
    assert shown()["Printer / nozzle"].endswith("0.4 mm nozzle") and shown()["Layer height"] == "0.12 mm"
    assert "three times" in win.nozzle_note.text() and "Fit the 0.2" not in win.nozzle_note.text()
    assert not win.dirty

    win.nozzle_seg.button("0.2").click()
    settle(app, win)
    assert win.params.nozzle == "0.2" and win.dirty and win._error is None
    assert shown()["Printer / nozzle"].endswith("0.2 mm nozzle")
    assert shown()["Process preset"] == "0.06mm High Quality @BBL A1 0.2 nozzle"
    assert shown()["Layer height"] == "0.06 mm" and shown()["Wall loops"] == "11"
    assert "Fit the 0.2 mm nozzle" in win.nozzle_note.text()
    assert "Ready to print" in win.verdict.text()
    assert "0.2 mm nozzle" in details() and "0.06 mm layers" in details()

    path = tmp_path / "session.json"
    last_dir = win._last_dir
    try:
        win.session_path = str(path)
        win._save_session()
        assert json.loads(path.read_text(encoding="utf-8"))["params"]["nozzle"] == "0.2"
        win._new_project(confirm=False)
        settle(app, win)
        assert win.params.nozzle == "0.4" and win.nozzle_seg.current() == "0.4"
        assert shown()["Layer height"] == "0.12 mm"
        assert win._load_file(str(path), as_session=True)
        settle(app, win)
        assert win.params.nozzle == "0.2" and win.nozzle_seg.current() == "0.2"
        assert shown()["Layer height"] == "0.06 mm"
    finally:
        win.session_path = None
        win._last_dir = last_dir
    win.nozzle_seg.button("0.4").click()
    settle(app, win)
    assert win.params.nozzle == "0.4" and shown()["Wall loops"] == "6"


# ---------------------------------------------------------------------------
# tips
# ---------------------------------------------------------------------------
def _tip_of(widget) -> str:
    """The tooltip a widget shows: its own, or that of the nearest thing it sits in."""
    while widget is not None:
        if widget.toolTip():
            return widget.toolTip()
        widget = widget.parentWidget()
    return ""


def test_every_control_explains_itself(app, win):
    """Resting the mouse on anything that can be clicked, dragged or typed into says what it
    does: every button, list, slider, number box and text box of the window, every tab, every
    menu entry, every choice of a drop-down list and every line of the checklist."""
    from PyQt6 import QtCore, QtGui, QtWidgets
    from lithosphere.ui import tips
    from lithosphere.ui.widgets import CheckRow, ImageCard, ParamSlider

    kinds = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QAbstractSlider,
             QtWidgets.QAbstractSpinBox, QtWidgets.QPlainTextEdit, ParamSlider, ImageCard, CheckRow)
    silent, seen = [], 0
    for widget in win.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, kinds) or isinstance(widget, QtWidgets.QScrollBar):
            continue
        if widget.objectName().startswith("qt_") or isinstance(widget.parentWidget(), QtWidgets.QTabBar):
            continue                                    # Qt's own scroll and overflow buttons
        if widget.window() is not win and not isinstance(widget.window(), QtWidgets.QFrame):
            continue                                    # a message box left over from another test
        seen += 1
        tip = _tip_of(widget)
        if not tip.startswith("<table"):
            label = widget.text() if hasattr(widget, "text") else ""
            silent.append(f"{type(widget).__name__} {label!r} {tip[:30]!r}")
    assert seen > 120 and not silent, silent

    for index in range(win.tabs.count()):
        assert win.tabs.tabToolTip(index).startswith("<table"), win.tabs.tabText(index)
    menus = [top.menu() for top in win.menuBar().actions()]        # File, View, Help
    entries = [a for menu in menus for a in menu.actions() if not a.isSeparator()]
    bare = [(a.text(), a.toolTip()[:30]) for a in entries if not a.toolTip().startswith("<table")]
    assert len(menus) == 3 and len(entries) == 14 and not bare, bare
    assert all(menu.toolTipsVisible() for menu in menus)
    lists = [win.light_combo_preset, win.hanger_combo, win.quality_combo, *win.deco_combos.values()]
    for combo in lists:
        for index in range(combo.count()):
            tip = combo.itemData(index, QtCore.Qt.ItemDataRole.ToolTipRole)
            assert tip and tip.startswith("<table"), (combo.itemText(index), tip)
    for card in win.cards:
        assert f"Picture {card.index + 1}" in card.toolTip()
        assert "Choose photo" in card.choose_btn.toolTip() and "Remove this photo" in card.remove_btn.toolTip()
    rows = win.findChildren(CheckRow)
    assert rows and all(row.toolTip().startswith("<table") for row in rows)

    # a slider's tip ends with where it starts and how to get back there
    assert "Starts at 80 mm" in win.sliders["diameter"].toolTip()
    assert "Double-click the name" in win.sliders["gap_deg"].toolTip()
    assert "Starts at 85 %" in win.slot_sliders["text_size"].toolTip()
    # tips are short formatted boxes of one fixed width, with their title first
    tip = win.sliders["t_min"].toolTip()
    assert f'width="{tips.WIDTH_PX}"' in tip and "<b>" in tip and "Thinnest wall" in tip
    assert all(len(text) < 420 for entry in tips.TIPS.values() for text in entry)
    # every text is used somewhere; these few belong to boxes and states not open right now
    assert tips.unused() <= {"box.open", "box.folder", "guide.copy", "check.warn", "check.error"}
    # the settings table explains each row and says it is already in the project
    labels = [win.settings_grid.itemAt(i).widget() for i in range(win.settings_grid.count())]
    assert labels and all("already inside the ready-to-print project" in label.toolTip() for label in labels)
    assert any("Wall loops" in label.toolTip() and "Set it to: 6" in label.toolTip() for label in labels)


def test_tips_can_be_switched_off_and_the_choice_is_remembered(app, win, tmp_path):
    import json

    from PyQt6 import QtCore, QtGui

    def reaches(widget) -> bool:                        # would a tooltip request get through?
        event = QtGui.QHelpEvent(QtCore.QEvent.Type.ToolTip, QtCore.QPoint(5, 5),
                                 widget.mapToGlobal(QtCore.QPoint(5, 5)))
        return not win._tip_switch.eventFilter(widget, event)

    def other_events_pass() -> bool:
        return not win._tip_switch.eventFilter(win.export_btn, QtCore.QEvent(QtCore.QEvent.Type.Enter))

    path = tmp_path / "session.json"
    try:
        assert win.tips_action.isChecked() and reaches(win.export_btn) and other_events_pass()
        win.tips_action.trigger()                       # Help > Show tips
        assert not win.tips_action.isChecked() and not reaches(win.export_btn) and other_events_pass()
        win.session_path = str(path)
        win._save_session()
        assert json.loads(path.read_text(encoding="utf-8"))["tips"] is False
        win.tips_action.trigger()
        assert reaches(win.sliders["diameter"])
        assert win._load_file(str(path), as_session=True)
        assert not win.tips_action.isChecked() and not reaches(win.sliders["diameter"])
    finally:
        win.session_path = None
        win._set_tips(True)
    assert reaches(win.export_btn)


# ---------------------------------------------------------------------------
# lettering and decorations
# ---------------------------------------------------------------------------
def _pick(combo, key: str) -> None:
    """Choose an entry of a drop-down list the way a click does."""
    index = combo.findData(key)
    assert index >= 0, key
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def _link(win, text: str):
    from PyQt6 import QtWidgets

    found = [b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == text]
    assert len(found) == 1, text
    return found[0]


def _titles(win) -> list[str]:
    return [title for _level, title, _detail in win._checks_shown]


def test_lettering_is_typed_on_the_decorate_page_and_shows_on_the_ornament(app, win):
    from lithosphere import fonts

    win.tabs.setCurrentIndex(1)
    slot = win.params.images[0]
    assert "No lettering" in win.text_info.text() and not win.dirty
    assert not win.slot_sliders["text_size"].isEnabled() and not win.tone_seg.isEnabled()
    assert win.font_combo.currentData() == fonts.default_font()
    picks = fonts.festive()
    if picks:                                           # the festive fonts lead the list
        assert [win.font_combo.itemData(i) for i in range(len(picks))] == picks
    before = picture_bytes(win.preview.canvas)

    win.text_edit.setPlainText("Happy Holidays")
    settle(app, win)
    assert slot.text == "Happy Holidays" and win.dirty and win._error is None
    assert win.slot_sliders["text_size"].isEnabled() and win.tone_seg.isEnabled()
    assert "mm tall" in win.text_info.text() and fonts.default_font() in win.text_info.text()
    assert "Lettering" in _titles(win) and picture_bytes(win.preview.canvas) != before
    assert win.text_title.text() == "LETTERING ON PICTURE 1"

    win.slot_sliders["text_size"].spin.setValue(40)     # the box reads per cent
    assert slot.text_size == pytest.approx(0.40)
    win.slot_sliders["text_y"].spin.setValue(0.5)
    win.slot_sliders["text_outline"].spin.setValue(0.5)
    win.slot_sliders["text_bold"].spin.setValue(0.2)
    assert (slot.text_y, slot.text_outline, slot.text_bold) == (0.5, 0.5, 0.2)
    win.tone_seg.button("dark").click()
    assert slot.text_tone == "dark"
    other = win.font_combo.itemData(1 if win.font_combo.count() > 1 else 0)
    _pick(win.font_combo, other)
    assert slot.text_font == other
    settle(app, win)
    assert other in win.text_info.text() and win._error is None

    # a long line comes out too small, and the note under the box says so at once
    win.text_edit.setPlainText("Happy Holidays from Johnny and Rachel and all the family 2026")
    win.slot_sliders["text_size"].spin.setValue(30)
    settle(app, win)
    assert "too small" in win.text_info.text() and "Lettering too small to print clearly" in _titles(win)
    assert "to check" in win.verdict.text()

    # the picture chosen here is the picture selected everywhere
    win.deco_pick.button("1").click()
    assert win.sel == 1 and win.text_edit.toPlainText() == "" and win.deco_pick.current() == "1"
    assert win.cards[1].property("selected") == "true" and win.text_title.text() == "LETTERING ON PICTURE 2"
    assert not win.slot_sliders["text_size"].isEnabled()
    win._select_slot(0)
    assert win.text_edit.toPlainText().startswith("Happy Holidays from") and win.deco_pick.current() == "0"
    assert win.tone_seg.current() == "dark" and win.font_combo.currentData() == other
    # only as many pictures can be picked as the ornament has
    assert [win.deco_pick.button(str(k)).isVisibleTo(win) for k in range(4)] == [True, True, True, False]

    # the note under the box always speaks of the picture that is selected
    assert "too small" in win.text_info.text()
    win.deco_pick.button("1").click()
    assert win.text_info.text() == "No lettering on this picture yet." and win.text_info.styleSheet() == ""
    win.text_edit.setPlainText("JOY")
    settle(app, win)
    win._select_slot(0)
    assert "too small" in win.text_info.text() and win.text_info.styleSheet() != ""
    win._select_slot(1)
    assert win.text_info.text().startswith("Letters ") and "too small" not in win.text_info.text()
    assert win.text_info.styleSheet() == ""

    # the box never holds more than is used: what does not fit is not kept on show
    win.text_edit.setPlainText("x" * 300)
    assert len(win.text_edit.toPlainText()) == 240 == len(win.params.images[1].text)


def test_a_picture_can_be_lettering_only(app, win, monkeypatch):
    from PyQt6 import QtWidgets

    win._select_slot(2)
    sample = win.params.images[2].path
    assert win.slot_sliders["zoom"].isEnabled() and not win.backdrop_seg.isEnabled()
    win._remove_photo(2)
    settle(app, win)
    slot = win.params.images[2]
    assert slot.path == "" and "Photo missing" in _titles(win)
    assert win.cards[2].name.text() == "no photo yet" and win.cards[2].thumb.text() == "drop\nphoto"
    assert not win.slot_sliders["zoom"].isEnabled() and not win.auto_levels.isEnabled()
    assert win.backdrop_seg.isEnabled() and "no photo" in win.deco_note.text()

    win.text_edit.setPlainText("Joy\nto the World")
    settle(app, win)
    assert "Photo missing" not in _titles(win) and "Lettering" in _titles(win)
    assert win.cards[2].name.text() == "lettering: Joy / to the World" and win.cards[2].thumb.text() == "Aa"
    assert "Ready to print" in win.verdict.text()
    plain = picture_bytes(win.preview.canvas)
    win.backdrop_seg.button("dark").click()
    settle(app, win)
    assert slot.backdrop == "dark" and picture_bytes(win.preview.canvas) != plain

    opened = []                                         # exporting does not ask about a placeholder
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (opened.append(1), ("", ""))[1]))
    win.asked.clear()
    win._export()
    assert opened and not any(a.startswith("question:") for a in win.asked)

    # a photo chosen for the picture goes under its lettering; taking it away again leaves the lettering
    win._assign_photos(2, [sample])
    slot = win.params.images[2]
    assert slot.path == os.path.abspath(sample) and slot.text == "Joy\nto the World" and slot.backdrop == "dark"
    assert not win.backdrop_seg.isEnabled() and win.slot_sliders["zoom"].isEnabled()
    assert win.cards[2].name.text() == os.path.basename(sample)
    win.slot_sliders["zoom"].spin.setValue(2.0)
    win._reset_slot()                                   # resets the photo, not what lies over it
    assert win.params.images[2].zoom == 1.0 and win.params.images[2].text == "Joy\nto the World"
    win._remove_photo(2)
    assert win.params.images[2].text == "Joy\nto the World" and win.params.images[2].path == ""
    _link(win, "Remove lettering and decorations from this picture").click()
    settle(app, win)
    assert not win.params.images[2].decorated and win.text_edit.toPlainText() == ""
    assert "Photo missing" in _titles(win) and win.cards[2].name.text() == "no photo yet"


def test_decorations_are_chosen_per_picture_and_can_be_copied_to_all(app, win):
    from lithosphere.params import DECO_SETTINGS

    win.tabs.setCurrentIndex(1)
    first, second = win.params.images[0], win.params.images[1]
    assert not win.slot_sliders["deco_amount"].isEnabled() and not win.shuffle_btn.isEnabled()
    assert not win.slot_sliders["deco_size"].isEnabled()
    before = picture_bytes(win.preview.canvas)
    _pick(win.deco_combos["deco_bottom"], "garland")
    settle(app, win)
    assert first.deco_bottom == "garland" and second.deco_bottom == "none"
    assert win.slot_sliders["deco_size"].isEnabled() and not win.shuffle_btn.isEnabled()
    with_garland = picture_bytes(win.preview.canvas)
    assert with_garland != before and win._error is None

    _pick(win.deco_combos["deco_scatter"], "snowflakes")
    assert win.slot_sliders["deco_amount"].isEnabled() and win.shuffle_btn.isEnabled()
    settle(app, win)
    snowy = picture_bytes(win.preview.canvas)
    win.shuffle_btn.click()
    settle(app, win)
    assert first.deco_seed == 1 and picture_bytes(win.preview.canvas) != snowy
    win.slot_sliders["deco_amount"].spin.setValue(90)
    win.slot_sliders["deco_size"].spin.setValue(1.3)
    assert first.deco_amount == pytest.approx(0.9) and first.deco_size == pytest.approx(1.3)
    _pick(win.deco_combos["deco_top"], "lights")
    _pick(win.deco_combos["deco_frame"], "line")
    first.text = "Only here"

    win._remove_photo(3)                                # a picture that is not in use, and empty
    _link(win, "Use these decorations on every picture").click()
    for slot in win.params.images[:3]:
        assert all(getattr(slot, name) == getattr(first, name) for name in DECO_SETTINGS)
    assert second.text == "" and second.path                     # lettering and photos are not copied
    assert not win.params.images[3].decorated           # ... nor are pictures that are not shown:
    win._set_count(4)                                   # an empty one must still say so
    settle(app, win)
    assert "Photo missing" in _titles(win) and win.cards[3].name.text() == "no photo yet"
    win._set_count(3)
    win._select_slot(0)
    settle(app, win)
    win._select_slot(1)
    assert {name: combo.currentData() for name, combo in win.deco_combos.items()} == {
        "deco_top": "lights", "deco_bottom": "garland", "deco_frame": "line", "deco_scatter": "snowflakes"}
    assert win.deco_title.text() == "DECORATIONS ON PICTURE 2"
    _link(win, "Remove lettering and decorations from this picture").click()
    assert not second.decorated and first.decorated and win.deco_combos["deco_top"].currentData() == "none"
    settle(app, win)
    assert win._error is None and "Ready to print" in win.verdict.text()


def test_lettering_and_decorations_survive_a_session_and_a_project(app, win, tmp_path):
    slot = win.params.images[1]
    slot.text, slot.text_tone, slot.text_size, slot.text_y = "Happy\nHolidays", "dark", 0.6, -0.9
    slot.deco_top, slot.deco_scatter, slot.deco_seed, slot.backdrop = "holly", "stars", 4, "dark"
    win.params.images[2] = type(slot)(text="2026", deco_frame="candy")       # lettering only
    win._sync_widgets()
    settle(app, win)
    before = win.params.to_dict()
    session, project = tmp_path / "session.json", tmp_path / "mine.lsphere"
    last_dir, project_path = win._last_dir, win.project_path
    try:
        win.session_path = str(session)
        win._save_session()
        assert win._write_project(str(project))
        for path in (session, project):
            win._new_project(confirm=False)
            assert not any(s.decorated for s in win.params.images)
            assert win._load_file(str(path), as_session=path is session)
            assert win.params.to_dict() == before
            win._select_slot(1)
            assert win.text_edit.toPlainText() == "Happy\nHolidays" and win.tone_seg.current() == "dark"
            assert win.deco_combos["deco_top"].currentData() == "holly"
            assert win.slot_sliders["text_size"].spin.value() == pytest.approx(60.0)
            assert win.cards[2].name.text() == "lettering: 2026"
            settle(app, win)
            assert win._error is None and "Photo missing" not in _titles(win)
    finally:
        win.session_path, win._last_dir, win.project_path = None, last_dir, project_path


def test_a_font_this_computer_lacks_is_shown_as_missing_until_another_is_chosen(app, win):
    from lithosphere import fonts

    slot = win.params.images[0]
    slot.text, slot.text_font = "Joy", "No Such Font Anywhere"
    count = win.font_combo.count()
    win._sync_widgets()
    settle(app, win)
    assert win.font_combo.count() == count + 1 and win.font_combo.currentIndex() == 0
    assert win.font_combo.currentText() == "No Such Font Anywhere  (not on this computer)"
    assert "Font not found" in _titles(win) and fonts.default_font() in win.text_info.text()
    win._select_slot(1)                                 # another picture: the stand-in goes
    assert win.font_combo.count() == count and win.font_combo.currentData() == fonts.default_font()
    win._select_slot(0)
    assert win.font_combo.count() == count + 1
    real = win.font_combo.itemData(1)
    _pick(win.font_combo, real)
    settle(app, win)
    assert slot.text_font == real and win.font_combo.count() == count
    assert win.font_combo.currentData() == real and "Font not found" not in _titles(win)


def test_the_rest_of_the_ball_is_decorated_at_the_foot_of_the_decorate_page(app, win, tmp_path):
    from PyQt6.QtCore import Qt
    from lithosphere.params import BALL_DIVIDERS, BALL_POLE, BALL_RING

    win.tabs.setCurrentIndex(1)
    combos = win.ball_combos
    assert list(combos) == ["ball_dividers", "ball_top", "ball_bottom", "ball_pole"]
    assert [c.currentData() for c in combos.values()] == ["none"] * 4
    assert [c.count() for c in combos.values()] == [len(BALL_DIVIDERS), len(BALL_RING),
                                                    len(BALL_RING), len(BALL_POLE)]
    assert "divider at the back" in win.ball_note.text() and win.ball_note.styleSheet() == ""
    assert "Decorations on the ball" not in _titles(win)
    before = picture_bytes(win.preview.canvas)
    win.dirty = False
    _pick(combos["ball_dividers"], "stars")
    settle(app, win)
    assert win.params.ball_dividers == "stars" and win.dirty
    assert picture_bytes(win.preview.canvas) != before and win._error is None
    _pick(combos["ball_bottom"], "snowflakes")
    _pick(combos["ball_pole"], "snowflake")
    settle(app, win)
    assert "Ready to print" in win.verdict.text()
    said = {title: detail for _level, title, detail in win._checks_shown}["Decorations on the ball"]
    assert said.startswith("Stars between the pictures, row of snowflakes round the bottom, big "
                           "snowflake underneath.")
    assert win.ball_note.styleSheet() == ""

    # one that finds no room says so under the lists and in the checklist; the rest stays
    _pick(combos["ball_top"], "fir")
    settle(app, win)
    assert "No room round the top" in win.ball_note.text() and win.ball_note.styleSheet() != ""
    assert "No room round the top" in _titles(win) and "thing to check" in win.verdict.text()
    assert "Decorations on the ball" in _titles(win)
    win.sliders["band_height_deg"].spin.setValue(96)            # as the checklist advises
    settle(app, win)
    assert "No room round the top" not in _titles(win) and win.ball_note.styleSheet() == ""
    assert "fir garland round the top" in {t: d for _l, t, d in win._checks_shown}[
        "Decorations on the ball"]

    # every list and every choice in it explains itself
    for combo in combos.values():
        assert "<table" in combo.toolTip()
        for index in range(combo.count()):
            assert "<table" in combo.itemData(index, Qt.ItemDataRole.ToolTipRole)

    # both wreaths can be chosen for a picture
    frames = win.deco_combos["deco_frame"]
    assert {frames.itemData(i) for i in range(frames.count())} >= {"wreath", "wreath_square", "candy"}
    _pick(frames, "wreath_square")
    assert win.params.images[win.sel].deco_frame == "wreath_square"

    # all of it is saved with the project, and a new ornament starts plain
    project = tmp_path / "ball.lsphere"
    project_path, last_dir = win.project_path, win._last_dir
    try:
        assert win._write_project(str(project))
        win._new_project(confirm=False)
        assert not win.params.ball_decorated
        assert [c.currentData() for c in combos.values()] == ["none"] * 4
        assert win._load_file(str(project), as_session=False)
        assert [c.currentData() for c in combos.values()] == ["stars", "fir", "snowflakes", "snowflake"]
        assert win.params.band_height_deg == 96.0 and win.params.images[0].deco_frame == "wreath_square"
        settle(app, win)
        assert win._error is None
    finally:
        win.project_path, win._last_dir = project_path, last_dir


def test_every_entry_of_a_decoration_list_is_explained_under_its_own_name(app, win):
    """Several entries share one explanation ("Row of stars" and "Stars"); each tooltip is
    still headed by the entry it belongs to, not by another entry's name."""
    from html import escape
    from PyQt6.QtCore import Qt

    seen = 0
    for combo in list(win.deco_combos.values()) + list(win.ball_combos.values()):
        for index in range(combo.count()):
            tip = combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
            name = escape(combo.itemText(index), quote=False)
            assert f">{name}</span></b>" in tip, combo.itemText(index)
            assert tip.count("<p ") >= 2                    # a heading and at least one paragraph
            seen += 1
    assert seen >= 35
