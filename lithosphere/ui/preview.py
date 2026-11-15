"""Interactive 3D preview of the ornament.

VTK draws the picture into an off-screen buffer and the widget shows it as an ordinary
image.  Nothing in this window ever presents an OpenGL frame to the screen, on purpose:
programs that draw overlays on 3D windows hook exactly that call, and a misbehaving one
takes the whole window down with it.  (Seen for real while testing this app: the Nahimic
sound overlay, which many motherboard audio drivers install, spins forever inside its
SwapBuffers hook when an OpenGL window sits on a second monitor.)  A plain image gives
such software nothing to hook, and it costs only a few milliseconds per frame.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
from PyQt6 import QtCore, QtGui, QtWidgets
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkPlane

from .. import render
from ..geometry import Body, Preview, body_normals, to_print_orientation
from ..params import BED_SIZE_MM, Params
from . import style

# key -> (button text, tooltip)
MODES = {
    "lit": ("Glowing", "How it looks on the tree with a light inside"),
    "unlit": ("Unlit", "The plain white print in daylight"),
    "cutaway": ("Cutaway", "Sliced in half to show the wall thickness"),
    "bed": ("On the printer", "Upside-down on the print bed, exactly as it is exported"),
}

PLASTIC = "#f3f2ee"
BACKGROUNDS = {
    "lit": ("#05070a", "#0e141a"),
    "unlit": ("#20252b", "#3b444d"),
    "cutaway": ("#20252b", "#3b444d"),
    "bed": ("#20252b", "#3b444d"),
}

MAX_ELEVATION = math.radians(88.0)     # the camera never goes over the top of the ornament
ZOOM_STEP = 1.1                        # per notch of the mouse wheel
ZOOM_RANGE = (0.62, 12.0)              # nearest / farthest camera distance, in ornament sizes
SPIN_DEG_PER_S = 18.0                  # "Spin": one turn in 20 seconds


def _pv_faces(faces: np.ndarray) -> np.ndarray:
    out = np.empty((len(faces), 4), dtype=np.int64)
    out[:, 0] = 3
    out[:, 1:] = faces
    return out.ravel()


def _zipper(pa: np.ndarray, pb: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate the strip between two polylines (indices: pa first, then pb)."""
    na, nb = len(pa), len(pb)
    i = j = 0
    tris = []
    while i < na - 1 or j < nb - 1:
        if j >= nb - 1:
            step_a = True
        elif i >= na - 1:
            step_a = False
        else:
            step_a = (np.sum((pa[i + 1] - pb[j]) ** 2) <= np.sum((pa[i] - pb[j + 1]) ** 2))
        if step_a:
            tris.append((i, i + 1, na + j))
            i += 1
        else:
            tris.append((i, na + j + 1, na + j))
            j += 1
    return tris


def section_mesh(body: Body, vertices: np.ndarray, column: int) -> pv.PolyData:
    """Flat cross-section of the wall on the plane through longitude columns
    `column` and the one opposite it (fills the cut made by the clipping plane)."""
    n_lon = body.n_lon
    points, tris, base = [], [], 0
    for col in (column % n_lon, (column + n_lon // 2) % n_lon):
        a = np.concatenate([[0], 1 + np.arange(body.outer_rows) * n_lon + col])
        b = body.n_outer + np.concatenate([[0], 1 + np.arange(body.inner_rows) * n_lon + col])
        pa, pb = vertices[a], vertices[b]
        tris += [(base + t0, base + t1, base + t2) for t0, t1, t2 in _zipper(pa, pb)]
        points += [pa, pb]
        base += len(pa) + len(pb)
    return pv.PolyData(np.concatenate(points).astype(np.float32),
                       _pv_faces(np.array(tris, dtype=np.int64)))


def _direction(lon: float, elevation: float) -> np.ndarray:
    """Unit vector from the ornament towards a camera at that longitude / elevation (radians)."""
    return np.array([math.cos(elevation) * math.cos(lon), math.cos(elevation) * math.sin(lon),
                     math.sin(elevation)])


class _Canvas(QtWidgets.QWidget):
    """Shows the rendered picture and reports what the mouse does on it."""

    dragged = QtCore.pyqtSignal(str, float, float)      # "turn" | "move" | "zoom", dx, dy in pixels
    wheeled = QtCore.pyqtSignal(float)                  # notches; positive = towards the ornament
    resized = QtCore.pyqtSignal()                       # size or screen scaling changed

    _BUTTONS = (QtCore.Qt.MouseButton.LeftButton | QtCore.Qt.MouseButton.MiddleButton
                | QtCore.Qt.MouseButton.RightButton)
    _SCALE_CHANGE = getattr(QtCore.QEvent.Type, "DevicePixelRatioChange", None)     # Qt 6.6+

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self._pixels = None                              # the image's memory; must outlive it
        self._fill = QtGui.QColor(BACKGROUNDS["lit"][0])
        self._drag: str | None = None
        self._last = QtCore.QPointF()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        # clicking the picture takes the focus, so a number still being typed in a box is applied
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(400, 400)

    def pixel_size(self) -> tuple[int, int]:
        """Size of the picture in real screen pixels."""
        dpr = self.devicePixelRatioF()
        return max(2, round(self.width() * dpr)), max(2, round(self.height() * dpr))

    def set_fill(self, color: str) -> None:
        self._fill = QtGui.QColor(color)

    def set_image(self, image: QtGui.QImage, pixels) -> None:
        self._image, self._pixels = image, pixels
        self.update()

    def image(self) -> QtGui.QImage | None:
        """The picture currently shown."""
        return self._image

    # -- painting -------------------------------------------------------------
    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self._fill)
        image = self._image
        if image is None:
            return
        dpr = image.devicePixelRatioF()
        if abs(dpr - self.devicePixelRatioF()) > 1e-6:   # drawn for another screen's scaling
            QtCore.QTimer.singleShot(0, self.resized.emit)
        if (abs(image.width() / dpr - self.width()) <= 1.0
                and abs(image.height() / dpr - self.height()) <= 1.0):
            painter.drawImage(QtCore.QPointF(0.0, 0.0), image)          # pixel for pixel
        else:                                # mid-resize: stretch until the new frame arrives
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(QtCore.QRectF(self.rect()), image)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.resized.emit()

    def event(self, event):
        if event.type() == self._SCALE_CHANGE:          # moved to a screen with other scaling
            self.resized.emit()
        return super().event(event)

    # -- mouse ----------------------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802
        button = event.button()
        shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if button == QtCore.Qt.MouseButton.LeftButton:
            kind = "move" if shift else "turn"
        elif button == QtCore.Qt.MouseButton.MiddleButton:
            kind = "move"
        elif button == QtCore.Qt.MouseButton.RightButton:
            kind = "zoom"
        else:
            super().mousePressEvent(event)
            return
        self._drag = kind
        self._last = event.position()
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag is None:
            return
        delta = event.position() - self._last
        self._last = event.position()
        if not delta.isNull():
            self.dragged.emit(self._drag, delta.x(), delta.y())
        event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag is not None and not (event.buttons() & self._BUTTONS):
            self._drag = None
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        event.accept()

    def wheelEvent(self, event):  # noqa: N802
        notches = event.angleDelta().y() / 120.0
        if notches:
            self.wheeled.emit(notches)
        event.accept()


class PreviewWidget(QtWidgets.QWidget):
    """The 3D view.  Call show_preview() with a fresh geometry.Preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = _Canvas(self)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        # Off-screen: VTK renders into its own buffer and never owns a visible window.
        self.plotter = pv.Plotter(off_screen=True, notebook=False,
                                  window_size=list(self.canvas.pixel_size()))
        try:
            self.plotter.enable_anti_aliasing("msaa")
        except Exception:        # not supported by every driver; purely cosmetic
            pass

        self._mode = "lit"
        self._light = "warm"
        self._preview: Preview | None = None
        self._params: Params | None = None
        self._mesh: pv.PolyData | None = None
        self._actor = None
        self._signature = None
        self._framed = False
        self._focus = 0                    # picture the user is working on
        self._alive = True                 # False once shutdown() has released the 3D view
        self._extent = 0.0                 # size the camera distance was last matched to
        self._queued = False               # a redraw is waiting for the event loop
        self._cut_plane = vtkPlane()
        self._spin = QtCore.QTimer(self)
        self._spin.setTimerType(QtCore.Qt.TimerType.PreciseTimer)    # coarse timers manage ~20 frames/s
        self._spin.setInterval(33)
        self._spin.timeout.connect(self._spin_step)
        self._spin_clock = QtCore.QElapsedTimer()
        self._apply_background()
        self.plotter.show(auto_close=False)          # off-screen: draws once and returns
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.wheeled.connect(self._on_wheel)
        self.canvas.resized.connect(self._queue_render)

    # -- public ---------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode == self._mode or mode not in MODES or not self._alive:
            return
        reframe = (mode == "bed") != (self._mode == "bed") or mode == "cutaway"
        self._mode = mode
        self._apply_background()
        self._refresh()
        if reframe and self._preview is not None:
            self.frame_default()

    def set_light(self, light: str) -> None:
        if light != self._light and self._alive:
            self._light = light
            self._refresh()

    def set_spinning(self, on: bool) -> None:
        if on and self._alive:
            self._spin_clock.start()
            self._spin.start()
        else:
            self._spin.stop()

    def show_preview(self, preview: Preview, params: Params) -> None:
        if not self._alive:
            return
        self._preview, self._params = preview, params
        self._refresh()
        if not self._framed:
            self._framed = True
            self.frame_default()
        elif abs(self._size() - self._extent) > 0.25:
            self._rescale_view()

    def frame_default(self) -> None:
        """Point the camera at a sensible starting view for the current mode."""
        if self._preview is None or not self._alive:
            return
        if self._mode == "cutaway":
            self.look_at(self._cut_longitude() - math.pi / 2, elevation=6.0)
        else:
            centers = self._preview.layout.centers
            self.look_at(centers[self._focus % len(centers)],
                         elevation=22.0 if self._mode == "bed" else 8.0)

    def look_at_picture(self, index: int) -> None:
        """Turn to face a picture.  In the cutaway the cut moves instead, so the
        slice goes through the middle of that picture and stays face-on."""
        self._focus = index
        if self._preview is None or not self._alive:
            return
        if self._mode == "cutaway":
            self._refresh()
            self.frame_default()
            return
        centers = self._preview.layout.centers
        self.look_at(centers[index % len(centers)])

    def look_at(self, lon: float, elevation: float = 8.0) -> None:
        """Place the camera facing the given longitude of the (hanging) ornament."""
        if self._preview is None or not self._alive:
            return
        if self._mode == "bed":
            lon = math.pi - lon                       # the model is flipped about Y
        center = self._center()
        self._extent = self._size()
        dist = 2.9 * self._extent
        cam = self.plotter.camera
        cam.focal_point = tuple(center)
        cam.position = tuple(center + dist * _direction(lon, math.radians(elevation)))
        cam.up = (0.0, 0.0, 1.0)
        cam.view_angle = 26.0
        self.plotter.reset_camera_clipping_range()
        self._render()

    def screenshot(self, path: str) -> None:
        self.plotter.screenshot(path)

    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False
        self._spin.stop()
        self.plotter.close()

    # -- drawing --------------------------------------------------------------
    def _render(self) -> None:
        """Draw the scene off-screen and put the finished picture on the canvas."""
        if not self._alive:
            return
        self._queued = False
        window = self.plotter.render_window
        size = self.canvas.pixel_size()
        if tuple(window.GetSize()) != size:
            window.SetSize(*size)
        self.plotter.render()
        width, height = window.GetSize()
        data = vtkUnsignedCharArray()
        window.GetRGBACharPixelData(0, 0, width - 1, height - 1, 0, data)
        if data.GetNumberOfTuples() != width * height:
            return                                     # nothing came back; keep the last picture
        # VTK's rows run bottom-up; a QImage's run top-down
        pixels = np.ascontiguousarray(vtk_to_numpy(data).reshape(height, width, 4)[::-1])
        image = QtGui.QImage(pixels.data, width, height, 4 * width, QtGui.QImage.Format.Format_RGBX8888)
        image.setDevicePixelRatio(self.canvas.devicePixelRatioF())
        self.canvas.set_image(image, pixels)

    def _queue_render(self) -> None:
        """Redraw once the mouse / resize events that are already waiting have been
        handled: they arrive faster than frames can be drawn."""
        if self._alive and not self._queued:
            self._queued = True
            QtCore.QTimer.singleShot(0, self._render_queued)

    def _render_queued(self) -> None:
        if self._queued:
            self._render()

    # -- mouse control --------------------------------------------------------
    def _on_drag(self, kind: str, dx: float, dy: float) -> None:
        if self._preview is None or not self._alive:
            return
        cam = self.plotter.camera
        focal = np.array(cam.focal_point, dtype=float)
        offset = np.array(cam.position, dtype=float) - focal
        dist = float(np.linalg.norm(offset))
        if dist <= 0.0:
            return
        width, height = max(1, self.canvas.width()), max(1, self.canvas.height())
        if kind == "turn":                 # the ornament follows the mouse; a full width = half a turn
            lon = math.atan2(offset[1], offset[0]) - dx / width * math.pi
            elevation = math.asin(max(-1.0, min(1.0, offset[2] / dist))) + dy / height * math.pi
            elevation = max(-MAX_ELEVATION, min(MAX_ELEVATION, elevation))
            cam.position = tuple(focal + dist * _direction(lon, elevation))
        elif kind == "move":               # slide the view; the ornament stays under the mouse
            view = -offset / dist
            right = np.cross(view, (0.0, 0.0, 1.0))
            length = float(np.linalg.norm(right))
            if length < 1e-9:
                return
            right /= length
            up = np.cross(right, view)
            per_pixel = 2.0 * dist * math.tan(math.radians(cam.view_angle) / 2.0) / height
            shift = (-dx * right + dy * up) * per_pixel
            cam.focal_point = tuple(focal + shift)
            cam.position = tuple(focal + shift + offset)
        else:                              # right-button drag: up = closer
            cam.position = tuple(focal + offset * (self._clamp_distance(dist * 1.01 ** dy) / dist))
        cam.up = (0.0, 0.0, 1.0)
        self.plotter.reset_camera_clipping_range()
        self._queue_render()

    def _on_wheel(self, notches: float) -> None:
        if self._preview is None or not self._alive:
            return
        cam = self.plotter.camera
        focal = np.array(cam.focal_point, dtype=float)
        offset = np.array(cam.position, dtype=float) - focal
        dist = float(np.linalg.norm(offset))
        if dist <= 0.0:
            return
        cam.position = tuple(focal + offset * (self._clamp_distance(dist / ZOOM_STEP ** notches) / dist))
        self.plotter.reset_camera_clipping_range()
        self._queue_render()

    def _clamp_distance(self, dist: float) -> float:
        size = self._size()
        return max(ZOOM_RANGE[0] * size, min(ZOOM_RANGE[1] * size, dist))

    # -- internals --------------------------------------------------------------
    def _size(self) -> float:
        lay = self._preview.layout
        return max(lay.height, lay.width)

    def _center(self) -> np.ndarray:
        lay = self._preview.layout
        if self._mode == "bed":
            return np.array([0.0, 0.0, lay.height / 2])
        return np.array([0.0, 0.0, (lay.z_top - lay.ro_frame) / 2])

    def _rescale_view(self) -> None:
        """The ornament changed size: keep the viewing direction and the user's
        zoom level, but move the camera so it fills the view as before."""
        cam = self.plotter.camera
        offset = np.array(cam.position) - np.array(cam.focal_point)
        new_extent = self._size()
        if self._extent > 0:
            offset *= new_extent / self._extent
        self._extent = new_extent
        center = self._center()
        cam.focal_point = tuple(center)
        cam.position = tuple(center + offset)
        self.plotter.reset_camera_clipping_range()
        self._render()

    def _apply_background(self) -> None:
        bottom, top = BACKGROUNDS[self._mode]
        self.plotter.set_background(bottom, top=top)
        self.canvas.set_fill(bottom)

    def _cut_longitude(self) -> float:
        """Longitude of the cutaway plane: through the middle of the picture in focus."""
        body = self._preview.body
        step = 2 * math.pi / body.n_lon
        centers = self._preview.layout.centers
        return round(centers[self._focus % len(centers)] / step) * step

    def _spin_step(self) -> None:
        if not self._alive:
            return
        seconds = min(0.1, self._spin_clock.restart() / 1000.0)      # same speed at any frame rate
        if not self.isVisible() or self.window().isMinimized():
            return                                     # nobody is watching: do not draw frames
        self.plotter.camera.Azimuth(SPIN_DEG_PER_S * seconds)
        self._render()

    def _refresh(self) -> None:
        prev, p = self._preview, self._params
        if prev is None or p is None:
            self._render()                             # nothing to show yet: just the background
            return
        pl = self.plotter
        body, lay = prev.body, prev.layout
        mode = self._mode
        bed = mode == "bed"
        points = to_print_orientation(body.vertices, lay) if bed else body.vertices
        points32 = np.ascontiguousarray(points, dtype=np.float32)

        # the triangles can be reused only if the grid is laid out exactly the same way
        signature = (len(body.vertices), len(body.faces), body.n_lon, body.n_outer,
                     body.outer_rows, body.inner_rows)
        fresh = self._mesh is None or signature != self._signature
        if fresh:
            self._mesh = pv.PolyData(points32, _pv_faces(body.faces))
            self._signature = signature
        else:
            self._mesh.points = points32
        mesh = self._mesh

        if mode == "lit":
            # coloured by the thickness the wall LOOKS when lit: relief on the inside of the
            # wall comes out blurred in a real print, and the preview shows it that way
            shown = body.lit if body.lit is not None else body.thickness
            mesh.point_data["glow"] = render.glow_rgb(shown, p.t_min, self._light)
        else:
            if "glow" not in mesh.point_data:
                mesh.point_data["glow"] = np.zeros((mesh.n_points, 3), dtype=np.uint8)
            mesh.point_data["Normals"] = body_normals(body, points)
            mesh.point_data.active_normals_name = "Normals"

        if fresh:
            self._actor = pl.add_mesh(mesh, name="body", scalars="glow", rgb=True, lighting=False,
                                      show_scalar_bar=False, reset_camera=False, render=False)
        prop, mapper = self._actor.prop, self._actor.mapper
        mapper.scalar_visibility = mode == "lit"
        prop.lighting = mode != "lit"
        if mode != "lit":
            prop.color = PLASTIC
            prop.interpolation = "Phong"
            prop.ambient, prop.diffuse = 0.18, 0.85
            prop.specular, prop.specular_power = 0.22, 24.0

        # --- cutaway ---------------------------------------------------------
        mapper.RemoveAllClippingPlanes()
        if mode == "cutaway":
            lon = self._cut_longitude()
            self._cut_plane.SetOrigin(0.0, 0.0, 0.0)
            self._cut_plane.SetNormal(-math.sin(lon), math.cos(lon), 0.0)
            mapper.AddClippingPlane(self._cut_plane)
            column = int(round(lon / (2 * math.pi / body.n_lon)))
            pl.add_mesh(section_mesh(body, body.vertices, column), name="section",
                        color=style.ACCENT, lighting=False, reset_camera=False, render=False)
        else:
            pl.remove_actor("section", render=False)

        # --- hanger loops ----------------------------------------------------
        if prev.hangers is not None:
            hv, hf = prev.hangers
            hv = to_print_orientation(hv, lay) if bed else hv
            hmesh = pv.PolyData(np.ascontiguousarray(hv, dtype=np.float32), _pv_faces(hf))
            if mode == "lit":
                tint = render.glow_rgb(np.array([p.cap_wall + 2.5]), p.t_min, self._light)[0]
                actor = pl.add_mesh(hmesh, name="hangers", color=[int(c) for c in tint],
                                    lighting=False, reset_camera=False, render=False)
            else:
                actor = pl.add_mesh(hmesh, name="hangers", color=PLASTIC, ambient=0.18,
                                    diffuse=0.85, specular=0.22, specular_power=24.0,
                                    reset_camera=False, render=False)
            if mode == "cutaway":
                actor.mapper.AddClippingPlane(self._cut_plane)
        else:
            pl.remove_actor("hangers", render=False)

        # --- print bed -------------------------------------------------------
        if bed:
            plate = pv.Plane(center=(0.0, 0.0, -0.15), direction=(0.0, 0.0, 1.0),
                             i_size=BED_SIZE_MM, j_size=BED_SIZE_MM, i_resolution=8, j_resolution=8)
            pl.add_mesh(plate, name="bed", color="#2d333a", show_edges=True, edge_color="#55606c",
                        lighting=False, reset_camera=False, render=False)
        else:
            pl.remove_actor("bed", render=False)

        pl.reset_camera_clipping_range()
        self._render()
