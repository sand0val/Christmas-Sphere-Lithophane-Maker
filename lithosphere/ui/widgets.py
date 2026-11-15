"""Reusable controls for the sidebar."""
from __future__ import annotations

import os
from html import escape

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from ..imaging import IMAGE_EXTENSIONS
from . import tips


def section(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text.upper())
    label.setObjectName("section")
    return label


def hint(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def rule() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setObjectName("rule")
    return line


class _WheelGuard(QtCore.QObject):
    """Stops the mouse wheel from changing a control.

    The sidebar scrolls, and a wheel turn that happens to pass over a slider
    or drop-down must scroll the page, not silently edit the ornament."""

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QtCore.QEvent.Type.Wheel:
            event.ignore()          # let the scroll area underneath have it
            return True
        return False


_wheel_guard = _WheelGuard()


def no_wheel(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Make a slider / number box / drop-down ignore the mouse wheel."""
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_wheel_guard)
    return widget


class ParamSlider(QtWidgets.QWidget):
    """A labelled slider with a linked number box.

    changed(value) fires for user edits only (never for set_value()).
    Double-click the label to go back to the default.

    tip names the entry of tips.TIPS that explains the setting.  scale is for
    show only: with scale=100 and suffix " %" a value of 0.7 reads "70 %".
    """

    changed = pyqtSignal(float)

    def __init__(self, label: str, lo: float, hi: float, step: float, value: float,
                 suffix: str = "", decimals: int = 1, tip: str = "", parent=None,
                 scale: float = 1.0):
        super().__init__(parent)
        self._lo, self._hi, self._step = lo, hi, step
        self._default = value
        self._scale = scale
        self._busy = False

        self.label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step * scale)
        self.spin.setRange(lo * scale, hi * scale)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        # typing a number outside the range snaps to the nearest allowed value
        self.spin.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setFixedWidth(92)
        self.spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(1, int(round((hi - lo) / step))))
        self.slider.setPageStep(max(1, self.slider.maximum() // 10))
        no_wheel(self.slider)
        no_wheel(self.spin)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.label, 1)
        top.addWidget(self.spin, 0)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 4)
        lay.setSpacing(3)
        lay.addLayout(top)
        lay.addWidget(self.slider)

        if tip:
            shown = f"{value * scale:.{decimals}f}{suffix}".strip()
            self.setToolTip(tips.tip(tip, foot=f"Starts at {shown}. Double-click the name to go "
                                               f"back to that."))
        self.label.installEventFilter(self)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value)

    # -- public ---------------------------------------------------------------
    def value(self) -> float:
        return float(self.spin.value()) / self._scale

    def set_value(self, value: float) -> None:
        """Set without emitting changed()."""
        self._busy = True
        try:
            value = min(self._hi, max(self._lo, float(value)))
            self.spin.setValue(value * self._scale)
            self.slider.setValue(int(round((value - self._lo) / self._step)))
        finally:
            self._busy = False

    def set_default(self, value: float) -> None:
        self._default = value

    def set_range(self, lo: float, hi: float) -> None:
        """Change the limits, keeping the value where possible (no signal)."""
        if hi <= lo:
            hi = lo + self._step
        current = self.value()
        self._busy = True
        try:
            self._lo, self._hi = lo, hi
            self.spin.setRange(lo * self._scale, hi * self._scale)
            self.slider.setRange(0, max(1, int(round((hi - lo) / self._step))))
        finally:
            self._busy = False
        self.set_value(current)

    # -- internals --------------------------------------------------------------
    def _from_slider(self, pos: int) -> None:
        if self._busy:
            return
        value = min(self._hi, self._lo + pos * self._step)
        self._busy = True
        try:
            self.spin.setValue(value * self._scale)
        finally:
            self._busy = False
        self.changed.emit(self.value())

    def _from_spin(self, value: float) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            self.slider.setValue(int(round((value / self._scale - self._lo) / self._step)))
        finally:
            self._busy = False
        self.changed.emit(self.value())

    def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
        if obj is self.label and event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            if abs(self.value() - self._default) > 1e-9:
                self.set_value(self._default)
                self.changed.emit(self.value())
            return True
        return super().eventFilter(obj, event)


class Segmented(QtWidgets.QWidget):
    """A row of mutually exclusive buttons."""

    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], current: str, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        for index, (key, text) in enumerate(options):
            button = QtWidgets.QPushButton(text)
            button.setObjectName("seg")
            pos = ("only" if len(options) == 1 else "first" if index == 0
                   else "last" if index == len(options) - 1 else "mid")
            button.setProperty("pos", pos)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(button)
            self._buttons[key] = button
            lay.addWidget(button, 1)
            button.clicked.connect(lambda _checked, k=key: self.changed.emit(k))
        self.set_current(current)

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def current(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def button(self, key: str) -> QtWidgets.QPushButton:
        return self._buttons[key]


class RadioChoice(QtWidgets.QWidget):
    """Vertical radio buttons, each with a one-line description."""

    changed = pyqtSignal(str)

    def __init__(self, options: dict[str, str], current: str, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._buttons: dict[str, QtWidgets.QRadioButton] = {}
        for key, text in options.items():
            button = QtWidgets.QRadioButton(text)
            self._buttons[key] = button
            lay.addWidget(button)
            button.clicked.connect(lambda _checked, k=key: self.changed.emit(k))
        self.set_current(current)

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def current(self) -> str:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return ""

    def set_option_enabled(self, key: str, enabled: bool) -> None:
        self._buttons[key].setEnabled(enabled)

    def button(self, key: str) -> QtWidgets.QRadioButton:
        return self._buttons[key]


class TipSwitch(QtCore.QObject):
    """Turns every tooltip of the application off and on again (Help menu)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.on = True

    def eventFilter(self, obj, event):  # noqa: N802
        return event.type() == QtCore.QEvent.Type.ToolTip and not self.on


def gray_pixmap(arr: np.ndarray) -> QtGui.QPixmap:
    """uint8 (h, w) array -> QPixmap."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape
    image = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(image.copy())


def _image_paths(mime: QtCore.QMimeData) -> list[str]:
    paths = []
    if mime.hasUrls():
        for url in mime.urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(path):
                paths.append(path)
    return paths


class ImageCard(QtWidgets.QFrame):
    """One picture slot: thumbnail, file name, choose / remove, drag-and-drop."""

    selected = pyqtSignal(int)
    choose = pyqtSignal(int)
    remove = pyqtSignal(int)
    dropped = pyqtSignal(int, list)

    THUMB = 64

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("card")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.thumb = QtWidgets.QLabel()
        self.thumb.setObjectName("thumb")
        self.thumb.setFixedSize(self.THUMB, self.THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title = QtWidgets.QLabel(f"Picture {index + 1}")
        self.title.setObjectName("cardTitle")
        self.name = QtWidgets.QLabel()
        self.name.setObjectName("hint")
        self.name.setTextFormat(Qt.TextFormat.PlainText)    # a file name or lettering, as typed
        self.name.setMinimumWidth(40)
        self.choose_btn = QtWidgets.QPushButton("Choose photo…")
        self.remove_btn = QtWidgets.QToolButton()
        self.remove_btn.setText("✕")
        self.setToolTip(tips.tip("card", n=index + 1))
        self.choose_btn.setToolTip(tips.tip("card.choose"))
        self.remove_btn.setToolTip(tips.tip("card.remove"))
        self.choose_btn.clicked.connect(lambda: self.choose.emit(self.index))
        self.remove_btn.clicked.connect(lambda: self.remove.emit(self.index))

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.choose_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addStretch(1)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(self.title)
        text.addWidget(self.name)
        text.addLayout(buttons)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)
        lay.addWidget(self.thumb)
        lay.addLayout(text, 1)
        self.set_selected(False)

    def set_photo(self, pixmap: QtGui.QPixmap | None, path: str, missing: bool = False,
                  note: str = "") -> None:
        """Show a photo, or (pixmap None) that there is none.  note says what a
        picture without a photo carries instead: its lettering, for example."""
        if pixmap is not None:
            self.thumb.setPixmap(pixmap.scaled(self.THUMB, self.THUMB,
                                               Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
        else:
            self.thumb.setPixmap(QtGui.QPixmap())
            self.thumb.setText("Aa" if note else "drop\nphoto")
        metrics = self.name.fontMetrics()
        if missing:
            self.name.setText("file not found")
        elif path:
            self.name.setText(metrics.elidedText(os.path.basename(path),
                                                 Qt.TextElideMode.ElideMiddle, 210))
        elif note:
            self.name.setText(metrics.elidedText(note, Qt.TextElideMode.ElideRight, 210))
        else:
            self.name.setText("no photo yet")
        self.name.setToolTip(path)
        self.remove_btn.setEnabled(bool(path))

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def _set_dragover(self, on: bool) -> None:
        self.setProperty("dragover", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    # -- events ---------------------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802
        self.selected.emit(self.index)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        if _image_paths(event.mimeData()):
            self._set_dragover(True)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):  # noqa: N802
        self._set_dragover(False)

    def dropEvent(self, event):  # noqa: N802
        self._set_dragover(False)
        paths = _image_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(self.index, paths)


class CheckRow(QtWidgets.QWidget):
    """One line of the print checklist: coloured mark, title, explanation."""

    def __init__(self, level: str, title: str, detail: str, color: str, icon: str, parent=None):
        super().__init__(parent)
        mark = QtWidgets.QLabel(icon)
        mark.setFixedSize(20, 20)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(f"background:{color}; color:#1b1f24; border-radius:10px; font-weight:700;")
        # (the detail can quote a font name or characters out of a project file: not markup)
        text = QtWidgets.QLabel(f"<b>{escape(title, quote=False)}</b><br>"
                                f"<span style='color:#9aa5b1'>{escape(detail, quote=False)}</span>")
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        self.setToolTip(tips.tip(f"check.{level}"))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 3, 0, 3)
        lay.setSpacing(10)
        lay.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(text, 1)
