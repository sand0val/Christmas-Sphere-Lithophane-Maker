"""Main window: controls on the left, live 3D preview on the right."""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import traceback

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from .. import (APP_NAME, __version__, checks, export, fonts, geometry, imaging, overlay,
                slicer)
from ..params import (BACKDROPS, BALL_DIVIDERS, BALL_POLE, BALL_RING, DECO_BOTTOM, DECO_FRAME,
                      DECO_SCATTER, DECO_SETTINGS, DECO_TOP,
                      DEFAULT_NOZZLE, HANGERS, LIGHT_PRESETS, LIMITS, MAX_IMAGES, MAX_TEXT_CHARS,
                      MIN_CAP_WALL, NOZZLES, PROJECTIONS, QUALITY_PRESETS, RELIEF_SIDES,
                      TEXT_TONES, GeometryError, ImageSlot, Params, cap_diameter_choices,
                      cap_diameter_range, make_buildable, max_band_height_deg, params_from_file,
                      seat_opening_max, shoulder_for_cap)
from . import style, tips
from .preview import MODES, PreviewWidget
from .widgets import (CheckRow, ImageCard, ParamSlider, RadioChoice, Segmented, TipSwitch,
                      gray_pixmap, hint, no_wheel, rule, section)

PROJECT_FILTER = "Lithophane sphere project (*.lsphere)"
SIDEBAR_WIDTH = 452
# the pages of the side panel, in the order of their tabs
PAGE_PICTURES, PAGE_DECORATE, PAGE_SPHERE, PAGE_TONES, PAGE_CAP, PAGE_PRINT = range(6)


def samples_in(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, "sample_*.png")))


class ExportWorker(QtCore.QThread):
    """Builds the full-detail model and writes it, off the GUI thread."""

    succeeded = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    stage = QtCore.pyqtSignal(str)

    def __init__(self, params: Params, path: str, make_project: bool, parent=None):
        super().__init__(parent)
        self._params, self._path, self._make_project = params, path, make_project

    def run(self) -> None:
        try:
            build = geometry.build_ornament(self._params)
            report = checks.verify_mesh(build.vertices, build.faces)
            if not (report["watertight"] and report["finite"] and report["volume_mm3"] > 0):
                raise RuntimeError(f"The generated mesh failed its integrity check: {report}")
            self.stage.emit("Writing the model…")
            export.write_mesh(self._path, build.vertices, build.faces, build.seam_faces)
            sheet_path = os.path.splitext(self._path)[0] + "_print_settings.txt"
            with open(sheet_path, "w", encoding="utf-8") as fh:
                fh.write(export.settings_sheet(self._params, build.layout, build.stats))
            info = {"path": self._path, "sheet": sheet_path, "stats": build.stats,
                    "bytes": os.path.getsize(self._path), "project": None, "slice": None,
                    "problems": list(build.problems),
                    "tips": len(checks.hanging_tips(build.vertices, build.faces))}
            if self._make_project:
                wait = ("this takes a minute or two with the 0.2 mm nozzle; "
                        if self._params.nozzle == "0.2" else "")
                self.stage.emit("Slicing with Bambu Studio to make the ready-to-print project…\n"
                                f"({wait}a Bambu Studio notice may flash up briefly)")
                project = slicer.project_path_for(self._path)
                try:
                    result = slicer.slice_mesh(build.vertices, build.faces, project, self._params,
                                               seam_faces=build.seam_faces)
                except Exception as exc:        # the model itself is fine; report and carry on
                    traceback.print_exc()
                    result = {"ok": False, "error": str(exc) or exc.__class__.__name__}
                info["slice"] = result
                if result.get("ok"):
                    info["project"] = project
            self.succeeded.emit(info)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


class _BusyDialog(QtWidgets.QProgressDialog):
    """A progress box that stays up until the work is done: Esc and the close
    button would only hide it while the export carried on unseen."""

    _done = False

    def finish(self) -> None:
        self._done = True
        self.close()

    def reject(self) -> None:
        if self._done:
            super().reject()

    def closeEvent(self, event):  # noqa: N802
        if self._done:
            super().closeEvent(event)
        else:
            event.ignore()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, samples_dir: str, session_path: str | None = None):
        super().__init__()
        self.samples_dir = samples_dir
        self.session_path = session_path
        self.params = Params()
        self.sel = 0                               # selected picture slot
        self.project_path: str | None = None
        self.dirty = False
        self.sliders: dict[str, ParamSlider] = {}
        self.slot_sliders: dict[str, ParamSlider] = {}
        self._preview_data: geometry.Preview | None = None
        self._error: str | None = None
        self._checks_shown: list[tuple] = []
        self._settings_shown: list[tuple] = []
        self._worker: ExportWorker | None = None
        self._closed = False
        self._last_dir = os.path.expanduser("~")
        self._make_project = True                  # also write a Bambu Studio project on export
        self._syncing = False                      # True while widgets are being set from params
        self._font_stand_in = False                # the font list starts with a missing font
        self._tip_switch = TipSwitch(self)         # Help > Show tips
        QtWidgets.QApplication.instance().installEventFilter(self._tip_switch)

        self.setAcceptDrops(True)
        self.setMinimumSize(1000, 640)
        screen = QtWidgets.QApplication.primaryScreen()
        room = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1920, 1040)
        self.resize(max(1000, min(1360, room.width() - 40)), max(640, min(880, room.height() - 60)))
        self._cap_ratio = (0.325, 0.1125)          # cap diameter / height as a share of the ball

        # Two-stage preview: a quick rebuild while a control is moving, then a
        # sharper one as soon as things have been still for a moment.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self._rebuild)
        self._sharp_timer = QtCore.QTimer(self)
        self._sharp_timer.setSingleShot(True)
        self._sharp_timer.setInterval(600)
        self._sharp_timer.timeout.connect(lambda: self._rebuild(sharp=True))

        self._build_ui()
        self._build_menu()
        if not (session_path and self._load_file(session_path, as_session=True)):
            self._new_project(confirm=False)
        self._update_title()

    # ======================================================================
    # UI construction
    # ======================================================================
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- sidebar -----------------------------------------------------------
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)

        header = QtWidgets.QVBoxLayout()
        header.setContentsMargins(18, 14, 18, 8)
        header.setSpacing(0)
        title = QtWidgets.QLabel(APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel("Photo ornaments for the Bambu Lab A1")
        subtitle.setObjectName("appSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        side.addLayout(header)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)
        for label, key, builder in (("1 Pictures", "pictures", self._page_pictures),
                                    ("2 Decorate", "decorate", self._page_decorate),
                                    ("3 Sphere", "sphere", self._page_sphere),
                                    ("4 Light && dark", "tones", self._page_tones),
                                    ("5 Cap", "cap", self._page_cap),
                                    ("6 Print", "print", self._page_print)):
            index = self.tabs.addTab(self._scrolled(builder()), label)
            self.tabs.setTabToolTip(index, tips.tip(f"tab.{key}"))
        side.addWidget(self.tabs, 1)

        footer = QtWidgets.QVBoxLayout()
        footer.setContentsMargins(18, 10, 18, 14)
        footer.setSpacing(8)
        self.export_btn = QtWidgets.QPushButton("Export for printing…")
        self.export_btn.setObjectName("primary")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setToolTip(tips.tip("export"))
        self.export_btn.clicked.connect(self._export)
        footer.addWidget(rule())
        footer.addWidget(self.export_btn)
        side.addLayout(footer)
        root.addWidget(sidebar)

        # --- right: view bar, preview, status ---------------------------------
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        viewbar = QtWidgets.QFrame()
        viewbar.setObjectName("viewbar")
        bar = QtWidgets.QHBoxLayout(viewbar)
        bar.setContentsMargins(12, 8, 12, 8)
        bar.setSpacing(10)
        self.mode_seg = Segmented([(k, v[0]) for k, v in MODES.items()], "lit")
        for key in MODES:
            self.mode_seg.button(key).setToolTip(tips.tip(f"mode.{key}"))
        self.mode_seg.setMaximumWidth(440)
        self.mode_seg.changed.connect(self._set_mode)
        bar.addWidget(self.mode_seg, 10)
        bar.addStretch(1)
        self.light_combo = QtWidgets.QComboBox()
        self.light_combo.addItem("Warm white light", "warm")
        self.light_combo.addItem("Cool white light", "cool")
        self.light_combo.setToolTip(tips.tip("light_colour"))
        self.light_combo.currentIndexChanged.connect(
            lambda _i: self.preview.set_light(self.light_combo.currentData()))
        bar.addWidget(no_wheel(self.light_combo))
        right.addWidget(viewbar)

        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.hide()
        banner_wrap = QtWidgets.QVBoxLayout()
        banner_wrap.setContentsMargins(12, 0, 12, 8)
        banner_wrap.addWidget(self.banner)
        self._banner_holder = QtWidgets.QFrame()
        self._banner_holder.setObjectName("viewbar")
        self._banner_holder.setLayout(banner_wrap)
        self._banner_holder.hide()
        right.addWidget(self._banner_holder)

        self.preview = PreviewWidget()
        right.addWidget(self.preview, 1)

        status = QtWidgets.QFrame()
        status.setObjectName("statusbar")
        srow = QtWidgets.QHBoxLayout(status)
        srow.setContentsMargins(14, 8, 14, 8)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("summary")
        self.summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                   QtWidgets.QSizePolicy.Policy.Preferred)     # never clipped
        self.summary.setToolTip(tips.tip("summary"))
        self.verdict = QtWidgets.QPushButton()
        self.verdict.setObjectName("link")
        self.verdict.setCursor(Qt.CursorShape.PointingHandCursor)
        self.verdict.setToolTip(tips.tip("verdict"))
        self.verdict.clicked.connect(lambda: self.tabs.setCurrentIndex(PAGE_PRINT))
        self.mouse_hint = QtWidgets.QLabel("Drag to turn  ·  scroll to zoom  ·  Shift-drag to move")
        self.mouse_hint.setObjectName("hint")
        srow.addWidget(self.summary)
        srow.addSpacing(16)
        srow.addWidget(self.verdict)
        srow.addStretch(1)
        srow.addWidget(self.mouse_hint)
        srow.addSpacing(10)
        self.look_label = QtWidgets.QLabel("Look at")
        srow.addWidget(self.look_label)
        self.face_buttons: list[QtWidgets.QToolButton] = []
        for k in range(MAX_IMAGES):
            button = QtWidgets.QToolButton()
            button.setText(str(k + 1))
            button.setToolTip(tips.tip("look", n=k + 1))
            button.clicked.connect(lambda _c, i=k: self._select_slot(i))
            self.face_buttons.append(button)
            srow.addWidget(button)
        self.spin_btn = QtWidgets.QToolButton()
        self.spin_btn.setText("Spin")
        self.spin_btn.setCheckable(True)
        self.spin_btn.setToolTip(tips.tip("spin"))
        self.spin_btn.toggled.connect(lambda on: self.preview.set_spinning(on))
        srow.addWidget(self.spin_btn)
        self.mouse_hint.setToolTip(tips.tip("mouse"))
        self.look_label.setToolTip(tips.tip("look", n=1).replace("picture 1", "a picture"))
        right.addWidget(status)

        holder = QtWidgets.QWidget()
        holder.setLayout(right)
        root.addWidget(holder, 1)

    def _scrolled(self, page: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        page.setObjectName("page")
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(page)
        return area

    @staticmethod
    def _page_layout(page: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(18, 12, 18, 16)
        lay.setSpacing(6)
        return lay

    def _slider(self, name: str, label: str, step: float, suffix: str,
                decimals: int) -> ParamSlider:
        """A slider for one setting of the ornament; tips.TIPS[name] explains it."""
        lo, hi = LIMITS[name]
        widget = ParamSlider(label, lo, hi, step, getattr(Params(), name), suffix, decimals, name)
        widget.changed.connect(lambda value, n=name: self._set_param(n, value))
        self.sliders[name] = widget
        return widget

    def _slot_slider(self, name: str, label: str, limit: str, step: float, suffix: str,
                     decimals: int, scale: float = 1.0) -> ParamSlider:
        """A slider for one setting of the selected picture; tips.TIPS[name] explains it."""
        lo, hi = LIMITS[limit]
        widget = ParamSlider(label, lo, hi, step, getattr(ImageSlot(), name), suffix, decimals,
                             name, scale=scale)
        widget.changed.connect(lambda value, n=name: self._set_slot_param(n, value))
        self.slot_sliders[name] = widget
        return widget

    # --- page 1: pictures ------------------------------------------------------
    def _page_pictures(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("How many pictures"))
        self.count_seg = Segmented([(str(n), str(n)) for n in range(1, MAX_IMAGES + 1)], "3")
        for n in range(1, MAX_IMAGES + 1):
            self.count_seg.button(str(n)).setToolTip(tips.tip(f"count.{n}"))
        self.count_seg.changed.connect(lambda key: self._set_count(int(key)))
        lay.addWidget(self.count_seg)
        lay.addWidget(hint("The pictures are spaced evenly around the sphere."))

        lay.addWidget(section("Photos"))
        self.cards: list[ImageCard] = []
        for k in range(MAX_IMAGES):
            card = ImageCard(k)
            card.selected.connect(self._select_slot)
            card.choose.connect(self._choose_photo)
            card.remove.connect(self._remove_photo)
            card.dropped.connect(self._drop_photos)
            self.cards.append(card)
            lay.addWidget(card)
        lay.addWidget(hint("Click a picture to adjust it. You can also drag photos here from "
                           "Explorer. Bright parts of a photo glow; dark parts stay dark. A "
                           "greeting and decorations are added on the Decorate page."))

        self.adjust_title = section("Adjust picture 1")
        lay.addWidget(self.adjust_title)
        self._photo_controls: list[QtWidgets.QWidget] = []

        def photo_control(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            self._photo_controls.append(widget)
            return widget

        lay.addWidget(photo_control(self._slot_slider("zoom", "Zoom", "zoom", 0.01, "x", 2)))
        lay.addWidget(photo_control(self._slot_slider("shift_x", "Move left / right", "shift",
                                                      0.01, "", 2)))
        lay.addWidget(photo_control(self._slot_slider("shift_y", "Move down / up", "shift",
                                                      0.01, "", 2)))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        for text, key, handler in (("⟲  Rotate left", "rotate_left", lambda: self._rotate(-90)),
                                   ("⟳  Rotate right", "rotate_right", lambda: self._rotate(90)),
                                   ("⇄  Mirror", "mirror", self._mirror)):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tips.tip(key))
            button.clicked.connect(handler)
            row.addWidget(photo_control(button))
        lay.addLayout(row)
        lay.addWidget(photo_control(self._slot_slider("brightness", "Brightness", "brightness",
                                                      0.01, "", 2)))
        lay.addWidget(photo_control(self._slot_slider("contrast", "Contrast", "contrast",
                                                      0.01, "x", 2)))
        self.auto_levels = QtWidgets.QCheckBox("Auto levels (use the full bright-to-dark range)")
        self.auto_levels.setToolTip(tips.tip("auto_levels"))
        self.auto_levels.toggled.connect(lambda on: self._set_slot_param("auto_levels", bool(on)))
        lay.addWidget(photo_control(self.auto_levels))
        reset = QtWidgets.QPushButton("Reset this picture's adjustments")
        reset.setObjectName("link")
        reset.setToolTip(tips.tip("reset_slot"))
        reset.clicked.connect(self._reset_slot)
        lay.addWidget(photo_control(reset))
        lay.addStretch(1)
        return page

    # --- page 2: decorate ------------------------------------------------------
    def _page_decorate(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("Which picture"))
        self.deco_pick = Segmented([(str(k), f"Picture {k + 1}") for k in range(MAX_IMAGES)], "0")
        for k in range(MAX_IMAGES):
            self.deco_pick.button(str(k)).setToolTip(tips.tip("deco.pick", n=k + 1))
        self.deco_pick.changed.connect(lambda key: self._select_slot(int(key)))
        lay.addWidget(self.deco_pick)
        self.deco_note = hint("")
        self.deco_note.setTextFormat(Qt.TextFormat.PlainText)       # it quotes a file name
        lay.addWidget(self.deco_note)

        # --- lettering --------------------------------------------------------
        self.text_title = section("Lettering on picture 1")
        lay.addWidget(self.text_title)
        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setObjectName("lettering")
        self.text_edit.setPlaceholderText("Type a greeting, for example\nHappy Holidays\n"
                                          "from all of us")
        self.text_edit.setTabChangesFocus(True)
        self.text_edit.setFixedHeight(self.text_edit.fontMetrics().lineSpacing() * 4 + 26)
        self.text_edit.setToolTip(tips.tip("text"))
        self.text_edit.textChanged.connect(self._text_edited)
        lay.addWidget(self.text_edit)
        self.text_info = hint("")
        self.text_info.setTextFormat(Qt.TextFormat.PlainText)       # it quotes a font name
        lay.addWidget(self.text_info)

        self._text_controls: list[QtWidgets.QWidget] = []

        def text_control(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            self._text_controls.append(widget)
            return widget

        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.setMaxVisibleItems(16)
        self.font_combo.setToolTip(tips.tip("font"))
        self._fill_fonts()
        self.font_combo.activated.connect(self._font_chosen)
        lay.addWidget(text_control(self._labelled("Font", no_wheel(self.font_combo))))
        self.tone_seg = Segmented(list(TEXT_TONES.items()), "light")
        for key in TEXT_TONES:
            self.tone_seg.button(key).setToolTip(tips.tip(f"tone.{key}"))
        self.tone_seg.changed.connect(lambda key: self._set_slot_param("text_tone", key))
        lay.addWidget(text_control(self.tone_seg))
        lay.addWidget(text_control(self._slot_slider("text_size", "Size", "text_size", 0.01,
                                                     " %", 0, scale=100.0)))
        lay.addWidget(text_control(self._slot_slider("text_y", "Move down / up", "shift",
                                                     0.01, "", 2)))
        lay.addWidget(text_control(self._slot_slider("text_x", "Move left / right", "shift",
                                                     0.01, "", 2)))
        lay.addWidget(text_control(self._slot_slider("text_outline", "Edge round the letters",
                                                     "text_outline", 0.05, " mm", 2)))
        lay.addWidget(text_control(self._slot_slider("text_bold", "Thicken letters",
                                                     "text_bold", 0.05, " mm", 2)))

        # --- decorations ------------------------------------------------------
        self.deco_title = section("Decorations on picture 1")
        lay.addWidget(self.deco_title)
        self.deco_combos: dict[str, QtWidgets.QComboBox] = {}
        for name, label, choices in (("deco_top", "Along the top", DECO_TOP),
                                     ("deco_bottom", "Along the bottom", DECO_BOTTOM),
                                     ("deco_frame", "All the way round", DECO_FRAME),
                                     ("deco_scatter", "Scattered", DECO_SCATTER)):
            combo = QtWidgets.QComboBox()
            for key, text in choices.items():
                combo.addItem(text, key)
                # (headed by the entry's own name: several entries share one explanation)
                combo.setItemData(combo.count() - 1,
                                  tips.rich(text, *tips.body(self._deco_tip(name, key))),
                                  Qt.ItemDataRole.ToolTipRole)
            combo.setToolTip(tips.tip(name))
            combo.activated.connect(
                lambda _i, n=name, c=combo: self._set_slot_param(n, c.currentData()))
            self.deco_combos[name] = combo
            lay.addWidget(self._labelled(label, no_wheel(combo)))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._slot_slider("deco_amount", "How many", "deco_amount", 0.01, " %", 0,
                                        scale=100.0), 1)
        self.shuffle_btn = QtWidgets.QPushButton("Shuffle")
        self.shuffle_btn.setToolTip(tips.tip("shuffle"))
        self.shuffle_btn.clicked.connect(self._shuffle)
        row.addWidget(self.shuffle_btn, 0, Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)
        lay.addWidget(self._slot_slider("deco_size", "Size of the decorations", "deco_size",
                                        0.05, "x", 2))

        # --- a picture without a photo ---------------------------------------
        self.backdrop_title = section("Picture without a photo")
        lay.addWidget(self.backdrop_title)
        self.backdrop_seg = Segmented(list(BACKDROPS.items()), "plain")
        for key in BACKDROPS:
            self.backdrop_seg.button(key).setToolTip(tips.tip(f"backdrop.{key}"))
        self.backdrop_seg.changed.connect(lambda key: self._set_slot_param("backdrop", key))
        lay.addWidget(self.backdrop_seg)
        self.backdrop_note = hint("")
        lay.addWidget(self.backdrop_note)

        lay.addSpacing(6)
        copy = QtWidgets.QPushButton("Use these decorations on every picture")
        copy.setObjectName("link")
        copy.setToolTip(tips.tip("copy_deco"))
        copy.clicked.connect(self._copy_decorations)
        lay.addWidget(copy)
        clear = QtWidgets.QPushButton("Remove lettering and decorations from this picture")
        clear.setObjectName("link")
        clear.setToolTip(tips.tip("clear_deco"))
        clear.clicked.connect(self._clear_decorations)
        lay.addWidget(clear)

        # --- the plain parts of the ball --------------------------------------
        lay.addWidget(section("The rest of the ball"))
        self.ball_combos: dict[str, QtWidgets.QComboBox] = {}
        for name, label, choices in (("ball_dividers", "Between pictures", BALL_DIVIDERS),
                                     ("ball_top", "Round the top", BALL_RING),
                                     ("ball_bottom", "Round the bottom", BALL_RING),
                                     ("ball_pole", "Underneath", BALL_POLE)):
            combo = QtWidgets.QComboBox()
            for key, text in choices.items():
                combo.addItem(text, key)
                combo.setItemData(combo.count() - 1, tips.rich(text, *tips.body(f"ball.{key}")),
                                  Qt.ItemDataRole.ToolTipRole)
            combo.setToolTip(tips.tip(name))
            combo.activated.connect(
                lambda _i, n=name, c=combo: self._set_param(n, c.currentData()))
            self.ball_combos[name] = combo
            lay.addWidget(self._labelled(label, no_wheel(combo)))
        self.ball_note = hint("")
        lay.addWidget(self.ball_note)
        lay.addStretch(1)
        return page

    @staticmethod
    def _labelled(label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """A control with its name in front of it, on one line."""
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(10)
        name = QtWidgets.QLabel(label)
        name.setFixedWidth(118)
        row.addWidget(name)
        row.addWidget(widget, 1)
        holder.setToolTip(widget.toolTip())
        return holder

    @staticmethod
    def _deco_tip(name: str, key: str) -> str:
        """The tips entry that describes one choice of a decoration list."""
        if name == "deco_scatter" and key == "snow":
            return "deco.snowdots"          # (the bank of snow along the bottom is "deco.snow")
        return f"deco.{key}"

    def _fill_fonts(self) -> None:
        """The festive fonts first, each shown in its own face, then every other
        font on this computer."""
        combo = self.font_combo
        combo.clear()
        families = set(QtGui.QFontDatabase.families())
        picks = fonts.festive()
        for name in picks:
            combo.addItem(name, name)
            face = self._face(name, families)
            if face is not None:
                combo.setItemData(combo.count() - 1, face, Qt.ItemDataRole.FontRole)
        if picks:
            combo.insertSeparator(combo.count())
        for name in fonts.installed():
            if name not in picks:
                combo.addItem(name, name)
        if combo.count() == 0:
            combo.addItem(fonts.BUILT_IN, fonts.BUILT_IN)

    @staticmethod
    def _face(name: str, families: set[str]) -> QtGui.QFont | None:
        """A font to show `name` in: its family is the longest beginning of the
        name that Qt knows, and what is left over says bold or italic."""
        words = name.split()
        for cut in range(len(words), 0, -1):
            family = " ".join(words[:cut])
            if family in families:
                rest = " ".join(words[cut:]).lower()
                face = QtGui.QFont(family)
                face.setPointSizeF(11.5)
                face.setBold("bold" in rest or "black" in rest)
                face.setItalic("italic" in rest or "oblique" in rest)
                return face
        return None

    # --- page 3: sphere --------------------------------------------------------
    def _page_sphere(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("Size"))
        lay.addWidget(self._slider("diameter", "Sphere diameter", 1.0, " mm", 0))
        lay.addWidget(section("Picture area"))
        lay.addWidget(self._slider("band_height_deg", "Picture height", 1.0, "°", 0))
        lay.addWidget(self._slider("gap_deg", "Gap between pictures", 0.5, "°", 1))
        self.panel_info = hint("")
        lay.addWidget(self.panel_info)

        lay.addWidget(section("How a flat photo is fitted to the ball"))
        self.projection = RadioChoice(PROJECTIONS, "natural")
        for key in PROJECTIONS:
            self.projection.button(key).setToolTip(tips.tip(f"projection.{key}"))
        self.projection.changed.connect(lambda key: self._set_param("projection", key))
        lay.addWidget(self.projection)
        self.projection_note = hint("")
        lay.addWidget(self.projection_note)

        lay.addWidget(section("Which side carries the picture"))
        self.relief = RadioChoice(RELIEF_SIDES, "outside")
        for key in RELIEF_SIDES:
            self.relief.button(key).setToolTip(tips.tip(f"relief.{key}"))
        self.relief.changed.connect(lambda key: self._set_param("relief_side", key))
        lay.addWidget(self.relief)
        lay.addWidget(hint(
            "Lit from inside, a picture on the outside of the wall looks sharp. A picture on the "
            "inside looks blurred: the light spreads sideways in the plastic after passing it, "
            "and small details wash out. Outside puts the whole picture on the outside. The "
            "ornament is printed upside-down, so in the half nearest the cap the outside faces "
            "down; where an edge there is too sudden to print, the inside of the wall carries "
            "part of it, and nothing hangs over thin air. Both puts the upper half of each "
            "picture inside the wall and the lower half outside; Inside puts all of it inside. "
            "That leaves the ornament smooth there, at the cost of a soft picture. The Glowing "
            "preview shows the difference."))
        self._add_reset(lay, ("diameter", "band_height_deg", "gap_deg", "projection", "relief_side"))
        lay.addStretch(1)
        return page

    # --- page 4: light & dark --------------------------------------------------
    def _page_tones(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("Wall thickness"))
        lay.addWidget(self._slider("t_min", "Thinnest wall (brightest)", 0.05, " mm", 2))
        lay.addWidget(self._slider("t_max", "Thickest wall (darkest)", 0.05, " mm", 2))
        lay.addWidget(self._slider("t_frame", "Frame thickness", 0.05, " mm", 2))
        lay.addWidget(section("Picture tone"))
        lay.addWidget(self._slider("gamma", "Mid-tone brightness", 0.01, "", 2))
        lay.addWidget(self._slider("smoothing", "Smoothing", 0.05, "", 2))
        self.safe_edges = QtWidgets.QCheckBox("Print-safe edges (recommended)")
        self.safe_edges.setToolTip(tips.tip("safe_edges"))
        self.safe_edges.toggled.connect(lambda on: self._set_param("safe_edges", bool(on)))
        lay.addWidget(self.safe_edges)
        self.invert = QtWidgets.QCheckBox("Negative (swap bright and dark)")
        self.invert.setToolTip(tips.tip("invert"))
        self.invert.toggled.connect(lambda on: self._set_param("invert", bool(on)))
        lay.addWidget(self.invert)
        lay.addWidget(hint("The Glowing preview simulates light passing through white PLA, so "
                           "what you see is close to the lit print."))
        self._add_reset(lay, ("t_min", "t_max", "t_frame", "gamma", "smoothing", "invert",
                              "safe_edges"))
        lay.addStretch(1)
        return page

    # --- page 5: cap, light, hanger --------------------------------------------
    def _page_cap(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("Light"))
        self.light_combo_preset = QtWidgets.QComboBox()
        for key, preset in LIGHT_PRESETS.items():
            self.light_combo_preset.addItem(preset.label, key)
            self.light_combo_preset.setItemData(self.light_combo_preset.count() - 1,
                                                tips.tip(f"light.{key}"),
                                                Qt.ItemDataRole.ToolTipRole)
        self.light_combo_preset.setToolTip(tips.tip("light_preset"))
        self.light_combo_preset.activated.connect(self._preset_chosen)
        lay.addWidget(no_wheel(self.light_combo_preset))
        lay.addWidget(self._slider("hole_diameter", "Light opening", 0.5, " mm", 1))
        self.seat_check = QtWidgets.QCheckBox("Seat for a light that sits on top of the cap")
        self.seat_check.setToolTip(tips.tip("seat"))
        self.seat_check.toggled.connect(lambda on: self._set_param("seat", bool(on)))
        lay.addWidget(self.seat_check)
        lay.addWidget(self._slider("seat_diameter", "Seat diameter", 0.1, " mm", 1))
        lay.addWidget(self._slider("seat_depth", "Seat depth", 0.5, " mm", 1))
        lay.addWidget(hint(
            "Use LED lights only - incandescent bulbs can soften PLA. A bulb of the tree's own "
            "string needs no batteries. The battery light that lasts is a micro-LED string on "
            "a 3 x AA box with a timer (one to three weeks of evenings per set): fold the lit "
            "end into a small bundle that hangs in the middle of the ball - an LED resting "
            "against a picture shows as a bright spot - and leave the box on a branch. A puck "
            "or tea light sits in a seat on top of the cap; coin cells keep a puck bright for "
            "hours, not weeks, and tea lights last because they are dim. Measure your light "
            "before printing; Custom size takes any diameter."))

        lay.addWidget(section("Cap"))
        lay.addWidget(self._slider("cap_diameter", "Cap diameter", 0.5, " mm", 1))
        lay.addWidget(self._slider("cap_height", "Cap height", 0.5, " mm", 1))
        lay.addWidget(self._slider("cap_wall", "Cap wall", 0.1, " mm", 1))
        lay.addWidget(self._slider("flare_angle", "Shoulder angle", 1.0, "°", 0))

        lay.addWidget(section("Hanger"))
        self.hanger_combo = QtWidgets.QComboBox()
        for key, label in HANGERS.items():
            self.hanger_combo.addItem(label, key)
            self.hanger_combo.setItemData(self.hanger_combo.count() - 1, tips.tip(f"hanger.{key}"),
                                          Qt.ItemDataRole.ToolTipRole)
        self.hanger_combo.setToolTip(tips.tip("hanger"))
        self.hanger_combo.activated.connect(
            lambda _i: self._set_param("hanger", self.hanger_combo.currentData()))
        lay.addWidget(no_wheel(self.hanger_combo))
        lay.addWidget(self._slider("hanger_hole", "Hanger hole", 0.1, " mm", 1))
        self._add_reset(lay, ("light_preset", "hole_diameter", "seat", "seat_diameter",
                              "seat_depth", "cap_diameter", "cap_height", "cap_wall",
                              "flare_angle", "hanger", "hanger_hole"))
        lay.addStretch(1)
        return page

    # --- page 6: print ---------------------------------------------------------
    def _page_print(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = self._page_layout(page)
        lay.addWidget(section("Nozzle"))
        self.nozzle_seg = Segmented([(key, nozzle.label) for key, nozzle in NOZZLES.items()],
                                    DEFAULT_NOZZLE)
        for key in NOZZLES:
            self.nozzle_seg.button(key).setToolTip(tips.tip(f"nozzle.{key}"))
        self.nozzle_seg.changed.connect(lambda key: self._set_param("nozzle", key))
        lay.addWidget(self.nozzle_seg)
        self.nozzle_note = hint("")
        lay.addWidget(self.nozzle_note)

        lay.addWidget(section("Model detail"))
        self.quality_combo = QtWidgets.QComboBox()
        for key, label in QUALITY_PRESETS.items():
            self.quality_combo.addItem(label, key)
            self.quality_combo.setItemData(self.quality_combo.count() - 1,
                                           tips.tip(f"quality.{key}"), Qt.ItemDataRole.ToolTipRole)
        self.quality_combo.setToolTip(tips.tip("quality"))
        self.quality_combo.activated.connect(
            lambda _i: self._set_param("quality", self.quality_combo.currentData()))
        lay.addWidget(no_wheel(self.quality_combo))

        self.project_check = QtWidgets.QCheckBox("Also make a ready-to-print Bambu Studio project")
        self.project_check.setToolTip(tips.tip("project_check"))
        self.project_check.setChecked(True)
        self.project_check.toggled.connect(self._set_make_project)
        if slicer.can_slice():
            lay.addSpacing(4)
            lay.addWidget(self.project_check)
        else:
            self.project_check.hide()

        lay.addWidget(section("Checklist"))
        self.checks_box = QtWidgets.QVBoxLayout()
        self.checks_box.setSpacing(0)
        lay.addLayout(self.checks_box)

        lay.addWidget(section("Bambu Studio settings"))
        lay.addWidget(hint("Already inside the ready-to-print project. Listed here in case you "
                           "slice the plain model yourself."))
        self.settings_grid = QtWidgets.QGridLayout()
        self.settings_grid.setHorizontalSpacing(12)
        self.settings_grid.setVerticalSpacing(4)
        self.settings_grid.setColumnStretch(1, 1)
        lay.addLayout(self.settings_grid)
        self.supports_note = hint(export.SUPPORTS_NOTE)
        lay.addSpacing(4)
        lay.addWidget(self.supports_note)
        guide = QtWidgets.QPushButton("Show the full printing guide…")
        guide.setObjectName("link")
        guide.setToolTip(tips.tip("guide"))
        guide.clicked.connect(self._show_guide)
        lay.addWidget(guide)
        lay.addStretch(1)
        return page

    def _set_make_project(self, on: bool) -> None:
        self._make_project = bool(on)

    def _add_reset(self, lay: QtWidgets.QVBoxLayout, names: tuple[str, ...]) -> None:
        button = QtWidgets.QPushButton("Reset this page to the recommended values")
        button.setObjectName("link")
        button.setToolTip(tips.tip("reset_page"))
        button.clicked.connect(lambda: self._reset_params(names))
        lay.addSpacing(6)
        lay.addWidget(button)

    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        def add(menu, text, handler, shortcut=None, tip=None):
            action = QtGui.QAction(text, self)
            if shortcut:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            if tip:
                action.setToolTip(tips.tip(tip))
            action.triggered.connect(handler)
            menu.addAction(action)
            return action

        file_menu.setToolTipsVisible(True)
        add(file_menu, "&New ornament", lambda: self._new_project(confirm=True), "Ctrl+N", "menu.new")
        add(file_menu, "&Open project…", self._open_project, "Ctrl+O", "menu.open")
        add(file_menu, "&Save project", self._save_project, "Ctrl+S", "menu.save")
        add(file_menu, "Save project &as…", self._save_project_as, "Ctrl+Shift+S", "menu.save_as")
        file_menu.addSeparator()
        add(file_menu, "&Export for printing…", self._export, "Ctrl+E", "export")
        file_menu.addSeparator()
        add(file_menu, "E&xit", self.close, "Ctrl+Q", "menu.exit")
        view_menu = bar.addMenu("&View")
        view_menu.setToolTipsVisible(True)
        for index, (key, (text, _tip)) in enumerate(MODES.items()):
            add(view_menu, text, lambda _c=False, k=key: self._set_mode(k), f"F{index + 1}",
                f"mode.{key}")
        view_menu.addSeparator()
        add(view_menu, "Reset view", lambda: self.preview.frame_default(), "Home",
            "menu.reset_view")
        help_menu = bar.addMenu("&Help")
        help_menu.setToolTipsVisible(True)
        add(help_menu, "Printing guide…", self._show_guide, "F12", "guide")
        self.tips_action = add(help_menu, "Show &tips when the mouse rests on something",
                               self._set_tips, tip="menu.tips")
        self.tips_action.setCheckable(True)
        self.tips_action.setChecked(True)
        help_menu.addSeparator()
        add(help_menu, "About", self._about, tip="menu.about")

    def _set_tips(self, on: bool) -> None:
        """Help > Show tips: every tooltip of the app on or off."""
        self._tip_switch.on = bool(on)
        self.tips_action.setChecked(bool(on))

    # ======================================================================
    # state changes
    # ======================================================================
    def _touch(self) -> None:
        self.dirty = True
        self._update_title()
        if not self._timer.isActive():      # throttle: a steady drag still redraws 40x a second
            self._timer.start()

    def _set_param(self, name: str, value) -> None:
        p = self.params
        before = getattr(p, name)
        setattr(p, name, value)
        if name == "diameter":
            self._scale_cap()
        elif name in ("cap_diameter", "cap_height"):
            self._remember_cap_ratio()
        if name == "t_min" and p.t_max < p.t_min + 0.5:
            p.t_max = min(LIMITS["t_max"][1], p.t_min + 0.5)
        elif name == "t_max" and p.t_min > p.t_max - 0.5:
            p.t_min = max(LIMITS["t_min"][0], p.t_max - 0.5)
        if name == "seat" and value:
            # a pocket needs a ledge under the light and a wall round it
            p.hole_diameter = self._seat_window(p.seat_diameter - 2 * self.SEAT_LEDGE)
            p.cap_diameter = max(p.cap_diameter, p.seat_diameter + 2 * self.SEAT_RIM)
        elif name == "seat_diameter" and p.seat:
            # the ledge under the light keeps its width, the wall round it its thickness
            p.hole_diameter = self._seat_window(p.hole_diameter + value - before)
            p.cap_diameter = max(p.cap_diameter, p.cap_diameter + value - before)
        self._fit_dependents()
        if name in ("seat", "seat_diameter") and p.seat:
            self._remember_cap_ratio()
        if name in ("hole_diameter", "seat", "seat_diameter", "seat_depth"):
            self._keep_or_drop_light_preset()
        self._sync_widgets()
        self._touch()

    SEAT_LEDGE = 2.5        # ledge left under a seated light when the app picks the opening
    SEAT_RIM = 2.5          # wall left round the seat when the app picks the cap

    @staticmethod
    def _seat_window(wanted: float) -> float:
        """A light opening of about `wanted` mm, on the slider's half-millimetre steps."""
        lo, hi = LIMITS["hole_diameter"]
        return min(hi, max(lo, math.floor(wanted * 2 + 1e-9) / 2))

    def _keep_or_drop_light_preset(self) -> None:
        """The light preset stays selected only while the cap is still made for
        that light; any other opening or seat is "Custom size"."""
        p = self.params
        preset = LIGHT_PRESETS[p.light_preset]
        same = (preset.hole is not None and abs(preset.hole - p.hole_diameter) < 1e-6
                and (preset.seat is not None) == bool(p.seat)
                and (preset.seat is None or (abs(preset.seat - p.seat_diameter) < 1e-6
                                             and abs(preset.depth - p.seat_depth) < 1e-6)))
        if not same:
            p.light_preset = "custom"

    def _remember_cap_ratio(self) -> None:
        p = self.params
        self._cap_ratio = (p.cap_diameter / p.diameter, p.cap_height / p.diameter)

    def _scale_cap(self) -> None:
        """Keep the cap in proportion to the ball, like a real bauble.  The
        proportion is remembered unrounded, so resizing back and forth returns
        to exactly the same cap."""
        p = self.params
        for name, share in zip(("cap_diameter", "cap_height"), self._cap_ratio):
            lo, hi = LIMITS[name]
            setattr(p, name, min(hi, max(lo, round(share * p.diameter * 2) / 2)))

    def _fit_dependents(self) -> None:
        """Keep linked values legal so ordinary edits never produce an error."""
        p = self.params
        _lo, hi = cap_diameter_range(p)
        hi = min(hi, LIMITS["cap_diameter"][1])        # the widest cap the app makes
        if p.seat:
            biggest_seat = math.floor((hi - 2 * MIN_CAP_WALL) * 10 + 1e-9) / 10
            if p.seat_diameter > biggest_seat:         # e.g. the sphere was made smaller
                if biggest_seat >= LIMITS["seat_diameter"][0]:
                    p.seat_diameter = biggest_seat
                else:
                    p.seat = False                     # no light would fit a seat that small
                p.light_preset = "custom"
        if p.seat:
            widest = self._seat_window(seat_opening_max(p))
            if p.hole_diameter > widest:
                p.hole_diameter = widest
                p.light_preset = "custom"
        biggest_opening = max(LIMITS["hole_diameter"][0], hi - 2 * MIN_CAP_WALL)
        if p.hole_diameter > biggest_opening:          # e.g. the sphere was made smaller
            p.hole_diameter = math.floor(biggest_opening * 2) / 2
            p.light_preset = "custom"
        lo, hi = cap_diameter_choices(p)
        hi = min(hi, LIMITS["cap_diameter"][1])
        if hi >= lo:
            p.cap_diameter = min(hi, max(lo, p.cap_diameter))
        p.band_height_deg = min(p.band_height_deg, max(LIMITS["band_height_deg"][0],
                                                        max_band_height_deg(p)))
        make_buildable(p)                   # the rarer rules (cap height, inner shoulder ...)
        p.band_height_deg = min(p.band_height_deg, max(LIMITS["band_height_deg"][0],
                                                        max_band_height_deg(p)))

    def _preset_chosen(self, _index: int) -> None:
        key = self.light_combo_preset.currentData()
        p = self.params
        preset = LIGHT_PRESETS[key]
        p.light_preset = key
        size = preset.hole
        shoulder_was = None
        if size is not None:
            p.hole_diameter = size
            p.seat = preset.seat is not None
            # make room for the opening first, then let the usual fitting clamp it
            if p.seat:
                p.seat_diameter, p.seat_depth = preset.seat, preset.depth
                # a light this wide may need a steeper shoulder, which leaves room for a wider cap
                roomy = preset.seat + 2 * self.SEAT_RIM
                angle = (shoulder_for_cap(p, roomy)
                         or shoulder_for_cap(p, preset.seat + 2 * MIN_CAP_WALL + 0.1))
                if angle is not None and angle > p.flare_angle:
                    shoulder_was, p.flare_angle = p.flare_angle, angle
                p.cap_diameter = max(p.cap_diameter, roomy)
            else:
                p.cap_diameter = max(p.cap_diameter, size + 2 * p.cap_wall + 4)
        self._fit_dependents()
        self._remember_cap_ratio()
        self._sync_widgets()
        self._touch()
        if preset.seat is not None and p.seat and p.seat_diameter >= preset.seat - 1e-6:
            if shoulder_was is not None:
                QtWidgets.QMessageBox.information(
                    self, "Shoulder made steeper",
                    f"A {preset.seat:g} mm seat needs a wider cap than a {p.diameter:.0f} mm sphere "
                    f"has room for with a {shoulder_was:.0f}\u00b0 shoulder, so the shoulder angle "
                    f"was raised to {p.flare_angle:.0f}\u00b0.\n\nA bigger sphere would make room "
                    f"as well.")
        elif preset.seat is not None:
            # (no shoulder angle makes a cap wide enough, or it would have been used)
            QtWidgets.QMessageBox.information(
                self, "Seat does not fit",
                f"A {preset.seat:g} mm seat does not fit the cap of a {p.diameter:.0f} mm sphere, "
                f"so " + (f"it was set to {p.seat_diameter:g} mm - the largest that fits."
                          if p.seat else "the seat was left out.")
                + "\n\nFor this light, make the sphere bigger first, then choose the light "
                  "again.")
        elif size is not None and self.params.hole_diameter < size - 1e-6:
            QtWidgets.QMessageBox.information(
                self, "Opening made smaller",
                f"A {size:g} mm opening does not fit a {self.params.diameter:.0f} mm sphere, so "
                f"it was set to {self.params.hole_diameter:g} mm - the largest that fits.\n\n"
                f"For this bulb, make the sphere bigger first.")

    def _reset_params(self, names: tuple[str, ...]) -> None:
        defaults = Params()
        for name in names:
            setattr(self.params, name, getattr(defaults, name))
        if "cap_diameter" in names:
            self._cap_ratio = (defaults.cap_diameter / defaults.diameter,
                               defaults.cap_height / defaults.diameter)
        if "diameter" in names or "cap_diameter" in names:
            self._scale_cap()               # the cap keeps (or returns to) its proportion
        self._fit_dependents()
        self._sync_widgets()
        self._touch()

    def _set_count(self, n: int) -> None:
        self.params.n_images = n
        self.sel = min(self.sel, n - 1)
        self._sync_widgets()
        self._touch()
        QtCore.QTimer.singleShot(60, lambda: self.preview.look_at_picture(self.sel))

    # --- picture slots ----------------------------------------------------------
    def _slot(self) -> ImageSlot:
        return self.params.images[self.sel]

    def _select_slot(self, index: int) -> None:
        if index >= self.params.n_images:
            return
        self.sel = index
        self._sync_slot_widgets()
        self.preview.look_at_picture(index)

    def _set_slot_param(self, name: str, value) -> None:
        setattr(self._slot(), name, value)
        self._refresh_card(self.sel)
        self._sync_slot_enabled()
        self._touch()

    # --- lettering and decorations (Decorate page) ----------------------------------
    def _text_edited(self) -> None:
        if self._syncing:
            return
        text = self.text_edit.toPlainText()
        if len(text) > MAX_TEXT_CHARS:          # the box never holds more than is used
            text = text[:MAX_TEXT_CHARS]
            self._syncing = True
            try:
                self.text_edit.setPlainText(text)
                self.text_edit.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            finally:
                self._syncing = False
        if text != self._slot().text:
            self._set_slot_param("text", text)

    def _font_chosen(self, _index: int) -> None:
        name = self.font_combo.currentData()
        self._set_slot_param("text_font", name)
        self._show_font(name)               # (drops a missing font from the top of the list)

    def _shuffle(self) -> None:
        """Deal the scattered shapes of the selected picture afresh."""
        self._set_slot_param("deco_seed", (self._slot().deco_seed + 1) % 10000)

    def _copy_decorations(self) -> None:
        """Give every other picture the decorations of the selected one."""
        source = self._slot()
        for k in range(self.params.n_images):       # (not the pictures that are not in use:
            if k != self.sel:                       # an empty one would stop counting as empty)
                for name in DECO_SETTINGS:
                    setattr(self.params.images[k], name, getattr(source, name))
        self._refresh_all_cards()
        self._touch()

    def _clear_decorations(self) -> None:
        self._slot().clear_overlay()
        self._sync_slot_widgets()
        self._refresh_card(self.sel)
        self._touch()

    def _rotate(self, degrees: int) -> None:
        slot = self._slot()
        # the mirror is applied after the turn, which reverses the turn's direction on screen
        slot.rotation = (slot.rotation + (-degrees if slot.flip else degrees)) % 360
        self._refresh_card(self.sel)
        self._touch()

    def _mirror(self) -> None:
        slot = self._slot()
        slot.flip = not slot.flip
        self._refresh_card(self.sel)
        self._touch()

    def _reset_slot(self) -> None:
        self._slot().reset_adjustments()
        self._sync_slot_widgets()
        self._refresh_card(self.sel)
        self._touch()

    def _choose_photo(self, index: int) -> None:
        start = os.path.dirname(self.params.images[index].path) or self._last_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Choose a photo for picture {index + 1}",
                                                        start, imaging.IMAGE_FILTER)
        if path:
            self._assign_photos(index, [path])

    def _remove_photo(self, index: int) -> None:
        # only the photo goes: its lettering and decorations belong to the picture
        self.params.images[index] = ImageSlot().keep_overlay_of(self.params.images[index])
        self._refresh_card(index)
        if index == self.sel:
            self._sync_slot_widgets()
        self._touch()

    def _drop_photos(self, index: int, paths: list) -> None:
        self._assign_photos(index, list(paths))

    def _assign_photos(self, first: int, paths: list[str]) -> None:
        """Put photos into consecutive slots starting at `first`, carrying on
        round to picture 1 when the last slot is reached."""
        bad, unused = [], []
        order = [(first + k) % MAX_IMAGES for k in range(MAX_IMAGES)]
        used: list[int] = []
        for path in paths:
            if len(used) >= MAX_IMAGES:
                unused.append(os.path.basename(path))
                continue
            try:
                imaging.load_photo(path)
            except Exception as exc:
                bad.append(f"{os.path.basename(path)}: {exc}")
                continue
            index = order[len(used)]
            self.params.images[index] = ImageSlot(path=os.path.abspath(path)).keep_overlay_of(
                self.params.images[index])
            self._last_dir = os.path.dirname(path)
            self._refresh_card(index)
            used.append(index)
        if used:
            # show every slot that just received a photo
            self.params.n_images = max(self.params.n_images, max(used) + 1)
            self.sel = first
            self._sync_widgets()
            self._touch()
            QtCore.QTimer.singleShot(60, lambda: self.preview.look_at_picture(first))
        notes = []
        if bad:
            notes.append("Could not read:\n" + "\n".join(bad))
        if unused:
            notes.append(f"Only {MAX_IMAGES} pictures fit on the ornament, so these were not "
                         f"used:\n" + "\n".join(unused))
        if notes:
            QtWidgets.QMessageBox.warning(self, "Not every photo was used", "\n\n".join(notes))

    # ======================================================================
    # widgets <- params
    # ======================================================================
    def _sync_widgets(self) -> None:
        p = self.params
        lo, hi = cap_diameter_choices(p)
        if hi >= lo:
            self.sliders["cap_diameter"].set_range(max(LIMITS["cap_diameter"][0], lo),
                                                   min(LIMITS["cap_diameter"][1], hi))
        _lo, hi = cap_diameter_range(p)
        hole_lo, hole_hi = LIMITS["hole_diameter"]
        hole_hi = min(hole_hi, math.floor((hi - 2 * MIN_CAP_WALL) * 2) / 2)
        if p.seat:
            hole_hi = min(hole_hi, self._seat_window(seat_opening_max(p)))
        self.sliders["hole_diameter"].set_range(hole_lo, max(hole_lo + 0.5, hole_hi))
        seat_lo, seat_hi = LIMITS["seat_diameter"]
        if p.seat:                          # (switched off, the slider just keeps its value)
            widest_cap = min(hi, LIMITS["cap_diameter"][1])
            seat_hi = min(seat_hi, math.floor((widest_cap - 2 * MIN_CAP_WALL) * 10 + 1e-9) / 10)
        self.sliders["seat_diameter"].set_range(seat_lo, max(seat_lo + 0.1, seat_hi))
        band_lo = LIMITS["band_height_deg"][0]
        self.sliders["band_height_deg"].set_range(band_lo, max(band_lo + 1, min(
            LIMITS["band_height_deg"][1], max_band_height_deg(p))))
        for name, widget in self.sliders.items():
            widget.set_value(getattr(p, name))
        self.count_seg.set_current(str(p.n_images))
        self.projection.set_current(p.projection)
        self.relief.set_current(p.relief_side)
        for combo, key in ((self.light_combo_preset, p.light_preset), (self.hanger_combo, p.hanger),
                           (self.quality_combo, p.quality)):
            combo.setCurrentIndex(max(0, combo.findData(key)))
        for box, value in ((self.invert, p.invert), (self.safe_edges, p.safe_edges),
                           (self.seat_check, p.seat)):
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)
        for name in ("seat_diameter", "seat_depth"):
            self.sliders[name].setEnabled(p.seat)
        self.nozzle_seg.set_current(p.nozzle)
        self.nozzle_note.setText(self._nozzle_text(p))
        self.sliders["hanger_hole"].setEnabled(p.hanger != "none")
        for k, card in enumerate(self.cards):
            card.setVisible(k < p.n_images)
            card.set_selected(k == self.sel)
        for k, button in enumerate(self.face_buttons):
            button.setVisible(k < p.n_images)
            self.deco_pick.button(str(k)).setVisible(k < p.n_images)
        for name, combo in self.ball_combos.items():
            combo.setCurrentIndex(max(0, combo.findData(getattr(p, name))))
        self._sync_slot_widgets()

    @staticmethod
    def _nozzle_text(p: Params) -> str:
        if p.nozzle == "0.2":
            return ("Fit the 0.2 mm nozzle before printing. The exported project already "
                    "selects the 0.2 mm printer, filament and process presets, with layers half "
                    "as tall and the model sampled finer to match. Expect about three times "
                    "the print time of the 0.4 mm nozzle.")
        return ("The nozzle the A1 comes with. The 0.2 mm nozzle shows finer detail in the "
                "picture, but takes about three times as long.")

    def _sync_slot_widgets(self) -> None:
        slot = self._slot()
        self._syncing = True
        try:
            for name, widget in self.slot_sliders.items():
                widget.set_value(getattr(slot, name))
            self.auto_levels.blockSignals(True)
            self.auto_levels.setChecked(slot.auto_levels)
            self.auto_levels.blockSignals(False)
            if self.text_edit.toPlainText() != slot.text:
                self.text_edit.setPlainText(slot.text)
            self._show_font(slot.text_font or fonts.default_font())
            self.tone_seg.set_current(slot.text_tone)
            self.backdrop_seg.set_current(slot.backdrop)
            for name, combo in self.deco_combos.items():
                combo.setCurrentIndex(max(0, combo.findData(getattr(slot, name))))
        finally:
            self._syncing = False
        number = self.sel + 1
        self.adjust_title.setText(f"ADJUST PICTURE {number}")
        self.text_title.setText(f"LETTERING ON PICTURE {number}")
        self.deco_title.setText(f"DECORATIONS ON PICTURE {number}")
        self.deco_pick.set_current(str(self.sel))
        for k, card in enumerate(self.cards):
            card.set_selected(k == self.sel)
        self._sync_slot_enabled()
        if self._preview_data is not None:          # the note under the text box is this picture's
            self._show_lettering_size(self._preview_data.layout)

    def _show_font(self, name: str) -> None:
        """Select a font in the list.  One this computer does not have is put at
        the top of the list, marked as missing, for as long as it is selected."""
        combo = self.font_combo
        if self._font_stand_in:
            combo.removeItem(0)
            self._font_stand_in = False
        index = combo.findData(name)
        if index < 0:
            combo.insertItem(0, f"{name}  (not on this computer)", name)
            self._font_stand_in = True
            index = 0
        combo.setCurrentIndex(index)

    def _sync_slot_enabled(self) -> None:
        """Grey out what cannot do anything for the selected picture as it is."""
        slot = self._slot()
        has_photo, has_text = bool(slot.path), bool(slot.text.strip())
        for widget in self._photo_controls:
            widget.setEnabled(has_photo)
        for widget in self._text_controls:
            widget.setEnabled(has_text)
        scattered = slot.deco_scatter != "none"
        self.slot_sliders["deco_amount"].setEnabled(scattered)
        self.shuffle_btn.setEnabled(scattered)
        self.slot_sliders["deco_size"].setEnabled(any(
            getattr(slot, name) != "none" for name in self.deco_combos))
        self.backdrop_seg.setEnabled(not has_photo)
        self.backdrop_title.setEnabled(not has_photo)
        number = self.sel + 1
        if has_photo:
            self.deco_note.setText(f"Picture {number} shows {os.path.basename(slot.path)}. "
                                   f"Lettering and decorations are laid over the photo.")
            self.backdrop_note.setText("Only for a picture that has no photo. Remove the photo "
                                       "on the Pictures page to make this one lettering only.")
        else:
            self.deco_note.setText(f"Picture {number} has no photo: it is a plain panel that "
                                   f"carries only lettering and decorations.")
            self.backdrop_note.setText("What the panel behind the lettering is made of. Plain "
                                       "blends into the ornament; Dark makes glowing letters "
                                       "stand out most; Bright glows all over.")

    def _refresh_card(self, index: int) -> None:
        slot = self.params.images[index]
        card = self.cards[index]
        if not slot.path:
            lines = overlay.text_lines(slot.text)
            note = (f"lettering: {' / '.join(lines)}" if lines
                    else "decorations only" if slot.decorated else "")
            card.set_photo(None, "", note=note)
            return
        try:
            photo = imaging.load_photo(slot.path)
            card.set_photo(gray_pixmap(imaging.thumbnail(photo, slot, 128)), slot.path)
        except Exception:
            card.set_photo(None, slot.path, missing=True)

    def _refresh_all_cards(self) -> None:
        for k in range(MAX_IMAGES):
            self._refresh_card(k)

    # ======================================================================
    # preview / status
    # ======================================================================
    def _rebuild(self, sharp: bool = False) -> None:
        p = self.params
        if self._closed:
            return                                  # a timer that outlived the window
        if sharp and self._timer.isActive():
            return                                  # more changes are on their way
        if sharp and QtWidgets.QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            self._sharp_timer.start()               # still dragging something: try again later
            return
        try:
            data = geometry.build_preview(p, sharp=sharp)
        except GeometryError as exc:
            self._set_error(str(exc))
            return
        except Exception as exc:            # a bug, not a user mistake: keep the app alive
            traceback.print_exc()
            self._set_error(f"Unexpected problem while building the preview: {exc}")
            return
        self._set_error(None)
        self._preview_data = data
        self.preview.show_preview(data, p)
        if sharp:
            return                                  # same design, just drawn finer
        self._update_status(data)
        self._sharp_timer.start()

    def _set_error(self, message: str | None) -> None:
        self._error = message
        self.banner.setText(message or "")
        self.banner.setVisible(bool(message))
        self._banner_holder.setVisible(bool(message))
        self.export_btn.setEnabled(not message)
        if message:
            self.verdict.setText("")
            self.summary.setText("Change the setting named above to continue.")

    def _update_status(self, data: geometry.Preview) -> None:
        p, lay = self.params, data.layout
        grams = data.volume_cm3 * 1.24
        pic_w = 2 * lay.panel_half * lay.R
        pic_h = 2 * lay.band_half * lay.R
        self.summary.setText(f"{lay.width:.0f} × {lay.height:.0f} mm  ·  {grams:.0f} g")
        plural = "Each picture" if p.n_images > 1 else "The picture"
        self.panel_info.setText(f"{plural} is about {pic_w:.0f} mm wide × {pic_h:.0f} mm tall on "
                                f"the surface.")
        forced = lay.projection != p.projection
        self.projection_note.setText("A single picture goes all the way round, so Wrap is used."
                                     if forced else "")
        self.projection_note.setVisible(forced)
        self._show_lettering_size(lay)
        self._show_ball_note(lay)

        results = checks.analyze(p, lay, data.volume_cm3, data.problems)
        level = checks.worst_level(results)
        n_warn = sum(1 for c in results if c.level != checks.OK)
        text = {"ok": "✓  Ready to print",
                "warn": f"!  {n_warn} thing{'s' if n_warn != 1 else ''} to check",
                "error": "✕  Cannot be printed as it is"}[level]
        self.verdict.setText(text)
        self.verdict.setStyleSheet(f"color: {style.LEVEL_COLORS[level]}; font-weight: 600;")

        shown = [(c.level, c.title, c.detail) for c in results]
        if shown != self._checks_shown:
            self._checks_shown = shown
            self._clear_layout(self.checks_box)
            ordered = sorted(results, key=lambda c: {"error": 0, "warn": 1, "ok": 2}[c.level])
            for c in ordered:
                self.checks_box.addWidget(CheckRow(c.level, c.title, c.detail,
                                                   style.LEVEL_COLORS[c.level],
                                                   style.LEVEL_ICONS[c.level]))
        self.supports_note.setText(export.supports_note(p))
        rows = export.slicer_settings(p)
        if rows != self._settings_shown:
            self._settings_shown = rows
            self._clear_layout(self.settings_grid)
            for r, (name, value, why) in enumerate(rows):
                left = QtWidgets.QLabel(name)
                left.setObjectName("value")
                right = QtWidgets.QLabel(value)
                right.setWordWrap(True)
                reason = [why[:1].upper() + why[1:] + "."] if why else []
                tip = tips.rich(name, f"Set it to: {value}.", *reason, *tips.body("settings"))
                left.setToolTip(tip)
                right.setToolTip(tip)
                self.settings_grid.addWidget(left, r, 0, Qt.AlignmentFlag.AlignTop)
                self.settings_grid.addWidget(right, r, 1)

    def _show_lettering_size(self, lay) -> None:
        """Under the text box: how big the lettering of the selected picture
        comes out, and whether the checklist has something to say about it."""
        slot = self._slot()
        text = overlay.lettering_for(slot, overlay.Panel.of(lay))
        if text is None:
            self.text_info.setText("No lettering on this picture yet.")
            self.text_info.setStyleSheet("")
            return
        note = (f"Letters {text.cap:.1f} mm tall, strokes about {text.stroke:.1f} mm wide, "
                f"in {text.font}.")
        if len(slot.text.strip().split("\n")) > overlay.MAX_LINES:
            note += f" Only the first {overlay.MAX_LINES} lines are used."
        flagged = [c for c in checks.lettering_checks(self.params, lay, only=self.sel)
                   if c.level != checks.OK]
        if flagged:
            note += f"  {style.LEVEL_ICONS[flagged[0].level]} {flagged[0].title} - see the " \
                    f"checklist on the Print page."
        self.text_info.setText(note)
        self.text_info.setStyleSheet(f"color: {style.LEVEL_COLORS[flagged[0].level]};"
                                     if flagged else "")

    def _show_ball_note(self, lay) -> None:
        """Under the lists for the plain parts of the ball: what they are for,
        or what the checklist has to say about the choices made."""
        flagged = [c for c in checks.ball_checks(self.params, lay) if c.level != checks.OK]
        if flagged:
            self.ball_note.setText(f"{style.LEVEL_ICONS[flagged[0].level]} {flagged[0].title} - "
                                   f"see the checklist on the Print page.")
            self.ball_note.setStyleSheet(f"color: {style.LEVEL_COLORS[flagged[0].level]};")
            return
        self.ball_note.setText("These go on the ball itself, between and around the pictures, "
                               "whatever the pictures show. The divider at the back always stays "
                               "plain: the print seam is hidden there.")
        self.ball_note.setStyleSheet("")

    @staticmethod
    def _clear_layout(layout: QtWidgets.QLayout) -> None:
        """Remove every widget from a layout at once.  (deleteLater alone leaves
        the old widgets painted where they were until the event loop gets to them.)"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _set_mode(self, mode: str) -> None:
        self.mode_seg.set_current(mode)
        self.light_combo.setEnabled(mode == "lit")
        if mode == self.preview.mode:
            self.preview.frame_default()            # clicking the current view again resets it
        else:
            self.preview.set_mode(mode)

    # ======================================================================
    # projects
    # ======================================================================
    def _update_title(self) -> None:
        name = os.path.basename(self.project_path) if self.project_path else "Untitled ornament"
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''}  -  {APP_NAME}")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Unsaved changes")
        box.setText("Save the changes to this ornament first?")
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Save
                               | QtWidgets.QMessageBox.StandardButton.Discard
                               | QtWidgets.QMessageBox.StandardButton.Cancel)
        answer = box.exec()
        box.deleteLater()
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return self._save_project()
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def _apply_params(self, params: Params) -> None:
        self.params = params
        self.sel = min(self.sel, params.n_images - 1)
        self._remember_cap_ratio()
        self._fit_dependents()
        self._sync_widgets()
        self._refresh_all_cards()
        self._timer.start()

    def _new_project(self, confirm: bool) -> None:
        if confirm and not self._confirm_discard():
            return
        params = Params()
        for k, path in enumerate(samples_in(self.samples_dir)[:MAX_IMAGES]):
            params.images[k].path = path
        self.project_path = None
        self.sel = 0
        self._apply_params(params)
        self.dirty = False
        self._update_title()

    def _load_file(self, path: str, as_session: bool = False) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            params = params_from_file(data)
        except Exception as exc:
            if not as_session:
                QtWidgets.QMessageBox.critical(self, "Cannot open project", f"{path}\n\n{exc}")
            return False
        for slot in params.images:
            # a built-in sample whose folder has moved since the file was saved
            if slot.path and not os.path.isfile(slot.path):
                moved = os.path.join(self.samples_dir, os.path.basename(slot.path))
                if os.path.basename(slot.path).startswith("sample_") and os.path.isfile(moved):
                    slot.path = moved
        if as_session:
            self.project_path = data.get("project_path") or None
            self._last_dir = data.get("last_dir") or self._last_dir
            self._make_project = bool(data.get("make_project", True))
            self.project_check.setChecked(self._make_project)
            self._set_tips(bool(data.get("tips", True)))
        else:
            self.project_path = path
            self._last_dir = os.path.dirname(path)
        self._apply_params(params)
        self.dirty = bool(data.get("dirty", False)) if as_session else False
        self._update_title()
        return True

    def _open_project(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", self._last_dir,
                                                        PROJECT_FILTER)
        if path:
            self._load_file(path)

    def _write_project(self, path: str) -> bool:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"app": APP_NAME, "version": __version__,
                           "params": self.params.to_dict()}, fh, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot save project", f"{path}\n\n{exc}")
            return False
        self.project_path = path
        self._last_dir = os.path.dirname(path)
        self.dirty = False
        self._update_title()
        return True

    def _save_project(self) -> bool:
        if not self.project_path:
            return self._save_project_as()
        return self._write_project(self.project_path)

    def _save_project_as(self) -> bool:
        start = self.project_path or os.path.join(self._last_dir, "my_ornament.lsphere")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", start, PROJECT_FILTER)
        if not path:
            return False
        if not path.lower().endswith(".lsphere"):
            path += ".lsphere"
        return self._write_project(path)

    def _save_session(self) -> None:
        if not self.session_path:
            return
        try:
            os.makedirs(os.path.dirname(self.session_path), exist_ok=True)
            with open(self.session_path, "w", encoding="utf-8") as fh:
                json.dump({"version": __version__, "params": self.params.to_dict(),
                           "project_path": self.project_path,
                           "last_dir": self._last_dir, "dirty": self.dirty,
                           "make_project": self._make_project,
                           "tips": self._tip_switch.on}, fh, indent=2)
        except OSError:
            pass                                    # losing the session is not worth an error box

    # ======================================================================
    # export
    # ======================================================================
    def _export(self) -> None:
        if self._error:
            QtWidgets.QMessageBox.warning(self, "Cannot export yet", self._error)
            return
        if self._worker is not None:
            return
        _photos, problems = imaging.photos_for(self.params)
        if problems:
            answer = QtWidgets.QMessageBox.question(
                self, "Picture without a photo",
                "\n".join(problems) + "\n\nA placeholder would be printed in its place. "
                "Export anyway?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentIndex(PAGE_PICTURES)
                return
        base = os.path.splitext(os.path.basename(self.project_path))[0] if self.project_path \
            else "lithophane_sphere"
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export for printing", os.path.join(self._last_dir, base + ".stl"),
            export.EXPORT_FILTER)
        if not path:
            return
        if not path.lower().endswith((".stl", ".3mf")):
            path += ".3mf" if "3mf" in chosen.lower() else ".stl"
        self._last_dir = os.path.dirname(path)

        self._progress = _BusyDialog("Building the full-detail model…", "", 0, 0, self)
        self._progress.setWindowTitle("Exporting")
        self._progress.setCancelButton(None)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setMinimumWidth(460)
        self._progress.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self._progress.show()
        make_project = self._make_project and slicer.can_slice()
        self._worker = ExportWorker(self.params.copy(), path, make_project, self)
        self._worker.stage.connect(self._progress.setLabelText)
        self._worker.succeeded.connect(self._export_done)
        self._worker.failed.connect(self._export_failed)
        self._worker.finished.connect(self._export_cleanup)
        self._worker.start()

    def _export_cleanup(self) -> None:
        self._progress.finish()
        self._worker = None

    def _export_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Export failed", message)

    def _export_done(self, info: dict) -> None:
        stats = info["stats"]
        sx, sy, sz = stats["size_mm"]
        model_line = (f"{sx:.0f} × {sy:.0f} × {sz:.0f} mm, about {stats['weight_g']:.0f} g "
                      f"of PLA, {stats['triangles']:,} triangles.")
        result, project = info.get("slice"), info.get("project")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Export complete")
        open_target = info["path"]
        if project:
            open_target = project
            box.setIcon(QtWidgets.QMessageBox.Icon.Information)
            box.setText(f"<b>{os.path.basename(project)}</b> is ready to print.")
            _expected, unexpected = slicer.split_warnings(result.get("warnings") or [])
            lines = [
                f"Print time about {slicer.format_duration(result.get('print_time_s'))}, "
                f"{result.get('filament_g') or 0:.0f} g of filament, {result.get('layer_count')} layers, "
                f"{'SUPPORTS WERE GENERATED' if result.get('support_used') else 'no supports'}.",
                "",
                "Open it in Bambu Studio, choose your white filament, press Slice plate, then "
                "Print plate. The printer, plate and every setting are already chosen and the "
                "ornament is the right way up.",
                "",
                export.SUPPORTS_NOTE,
            ]
            if self.params.nozzle == "0.2":
                lines[2:2] = ["This project is for the 0.2 mm nozzle: fit it, and change the "
                              "nozzle size on the printer's screen, before you print.", ""]
            seam_hidden, seam_text = slicer.seam_note(result.get("seam"))
            if seam_text:
                lines += ["", seam_text]
            rings_sound, rings_text = slicer.ring_note(result.get("seam"))
            if rings_text:
                lines += ["", rings_text]
            if unexpected or not seam_hidden or not rings_sound:
                box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            if unexpected:
                lines += ["", "Bambu Studio reported: " + "; ".join(unexpected)]
            lines += ["", f"Also saved: {os.path.basename(info['path'])} (plain model, {model_line}) "
                          f"and {os.path.basename(info['sheet'])}."]
            box.setInformativeText("\n".join(lines))
        else:
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning if result else
                        QtWidgets.QMessageBox.Icon.Information)
            box.setText(f"<b>{os.path.basename(info['path'])}</b> is ready to slice.")
            lines = [model_line, "",
                     "It is already turned the right way up for printing (cap on the bed). The "
                     f"print settings were saved next to it as {os.path.basename(info['sheet'])}."]
            if result:
                lines += ["", "The ready-to-print Bambu Studio project could not be made: "
                              f"{result.get('error') or 'unknown error'}",
                          "The model itself is fine - open it in Bambu Studio and apply the settings "
                          "from the Print page."]
            box.setInformativeText("\n".join(lines))
        if info.get("tips"):
            n = info["tips"]
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setInformativeText(
                box.informativeText() + f"\n\nWarning - {n} picture detail{'s' if n != 1 else ''} "
                f"would start in mid-air and print as {'blobs' if n != 1 else 'a blob'}. Switch "
                "Print-safe edges back on (Light & dark page), then export again.")
        if info.get("problems"):
            box.setInformativeText(box.informativeText() + "\n\nNote - a placeholder was printed "
                                   "instead of a photo:\n" + "\n".join(info["problems"]))
        open_btn = None
        if slicer.find_bambu_studio():
            open_btn = box.addButton("Open in Bambu Studio", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            open_btn.setToolTip(tips.tip("box.open"))
        folder_btn = box.addButton("Show in folder", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        folder_btn.setToolTip(tips.tip("box.folder"))
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()                   # (it is a child of the window: do not keep them all)
        if open_btn is not None and clicked is open_btn:
            slicer.open_in_bambu_studio(open_target)
        elif clicked is folder_btn:
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(open_target)])
            else:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(info["path"])))

    # ======================================================================
    # dialogs
    # ======================================================================
    def _show_guide(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Printing guide")
        dialog.resize(720, 560)
        lay = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        try:
            layout = self._preview_data.layout if self._preview_data else None
            sheet = export.settings_sheet(self.params, layout) if layout else "No model yet."
        except Exception as exc:
            sheet = str(exc)
        text.setPlainText(sheet)
        lay.addWidget(text)
        row = QtWidgets.QHBoxLayout()
        copy = QtWidgets.QPushButton("Copy to clipboard")
        copy.setToolTip(tips.tip("guide.copy"))
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text.toPlainText()))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        row.addWidget(copy)
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)
        dialog.exec()
        dialog.deleteLater()

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__}<br><br>Turns photos, greetings and Christmas "
            f"decorations into a lithophane bauble that prints in white PLA without supports "
            f"and glows when a tree light is pushed into its cap.")

    # ======================================================================
    # Qt events
    # ======================================================================
    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        # the mouse hint is the first thing to go when the window gets narrow
        hint_label = getattr(self, "mouse_hint", None)
        if hint_label is not None:
            hint_label.setVisible(self.width() >= 1342)
            self.look_label.setVisible(self.width() >= 1172)

    def dragEnterEvent(self, event):  # noqa: N802
        if any(u.toLocalFile().lower().endswith(imaging.IMAGE_EXTENSIONS)
               for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile().lower().endswith(imaging.IMAGE_EXTENSIONS)]
        if paths:
            event.acceptProposedAction()
            self._assign_photos(self.sel, paths)

    def closeEvent(self, event):  # noqa: N802
        if self._worker is not None:
            event.ignore()
            QtWidgets.QMessageBox.information(
                self, "Still exporting",
                "The ornament is still being exported. The window can be closed as soon as "
                "that has finished.")
            return
        self._closed = True
        self._timer.stop()
        self._sharp_timer.stop()
        self._save_session()
        self.preview.shutdown()
        event.accept()
