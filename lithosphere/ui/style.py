"""Look and feel: one dark theme with a warm 'glow' accent."""
from __future__ import annotations

import os
import tempfile

from PyQt6 import QtCore, QtGui, QtWidgets

BG = "#1b1f24"
PANEL = "#22272e"
RAISED = "#2c333b"
BORDER = "#3a434d"
TEXT = "#e8eaed"
MUTED = "#9aa5b1"
ACCENT = "#e9a23b"
ACCENT_HI = "#f6bb62"
ACCENT_TEXT = "#1b1f24"
DISABLED = "#66707b"
OK = "#5fbf77"
WARN = "#e9a23b"
ERROR = "#e5605a"

LEVEL_COLORS = {"ok": OK, "warn": WARN, "error": ERROR}
LEVEL_ICONS = {"ok": "✓", "warn": "!", "error": "✕"}

QSS = f"""
* {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; }}
QWidget#sidebar {{ background: {PANEL}; }}
QWidget#page {{ background: {PANEL}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QLabel#appTitle {{ font-size: 15pt; font-weight: 600; }}
QLabel#appSubtitle, QLabel#hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#section {{ color: {ACCENT}; font-weight: 600; font-size: 9pt; padding-top: 6px; }}
QLabel#value {{ color: {MUTED}; }}
QLabel#banner {{ background: {ERROR}; color: white; padding: 8px 12px; border-radius: 6px;
                 font-weight: 600; }}
QLabel#note {{ background: {RAISED}; color: {TEXT}; padding: 6px 10px; border-radius: 6px; }}
QLabel#summary {{ color: {TEXT}; font-size: 10pt; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; background: {PANEL}; }}
QTabBar {{ background: {PANEL}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 9px; margin: 0;
                border: none; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #66707b; border-color: #2f363e; }}
QPushButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT};
                       font-weight: 600; }}
QPushButton#primary {{ background: {ACCENT}; color: {ACCENT_TEXT}; border: none; font-weight: 700;
                       padding: 10px 14px; font-size: 11pt; }}
QPushButton#primary:hover {{ background: {ACCENT_HI}; }}
QPushButton#primary:disabled {{ background: #5a5140; color: #2a2a2a; }}
QPushButton#seg {{ border-radius: 0; padding: 6px 8px; margin: 0; border-right-width: 0; }}
QPushButton#seg[pos="first"] {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#seg[pos="last"] {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px;
                               border-right-width: 1px; }}
QPushButton#seg[pos="only"] {{ border-radius: 6px; border-right-width: 1px; }}
QPushButton#link {{ background: transparent; border: none; color: {ACCENT}; padding: 2px 0;
                    text-align: left; }}
QPushButton#link:hover {{ color: {ACCENT_HI}; text-decoration: underline; }}
QToolButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 9px; }}
QToolButton:hover {{ border-color: {ACCENT}; }}
QToolButton:disabled {{ color: #66707b; }}
QToolButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT};
                       font-weight: 600; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -6px 0;
                              border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HI}; }}
QSlider::handle:horizontal:disabled {{ background: #66707b; }}
QSlider::sub-page:horizontal:disabled {{ background: #5a5140; }}

QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{ background: {BG}; border: 1px solid {BORDER};
    border-radius: 5px; padding: 3px 6px; selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT}; }}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox {{ padding: 5px 8px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url("@ARROW@"); width: 10px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url("@ARROW_OFF@"); }}
QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{
    color: {DISABLED}; border-color: #2f363e; }}
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {{ color: {DISABLED}; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT}; outline: none; }}

QCheckBox, QRadioButton {{ spacing: 8px; padding: 3px 0; }}
QFrame#card {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#card[selected="true"] {{ border: 2px solid {ACCENT}; }}
QFrame#card[dragover="true"] {{ border: 2px dashed {ACCENT_HI}; }}
QLabel#thumb {{ background: {BG}; border-radius: 6px; color: {MUTED}; }}
QLabel#cardTitle {{ font-weight: 600; }}
QFrame#rule {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#viewbar, QFrame#statusbar {{ background: {PANEL}; }}

QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 6px; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: {RAISED}; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 16px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QScrollBar:vertical {{ background: {PANEL}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #55616e; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; height: 10px;
                text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QPlainTextEdit, QTextEdit {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px;
    font-family: Consolas, "Courier New", monospace; font-size: 9.5pt; }}
QPlainTextEdit#lettering {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt; padding: 4px; selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT}; }}
QPlainTextEdit#lettering:focus {{ border-color: {ACCENT}; }}
QPlainTextEdit#lettering:disabled {{ color: {DISABLED}; border-color: #2f363e; }}
"""


def _arrow_image(color: str, name: str) -> str:
    """Draw the little drop-down arrow once and return its path (style sheets
    can only take arrows from image files)."""
    folder = os.path.join(tempfile.gettempdir(), "lithophane_sphere_studio")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    scale = 4                                   # drawn large, shown small: stays crisp when scaled
    pixmap = QtGui.QPixmap(10 * scale, 6 * scale)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(0, 0), QtCore.QPointF(10 * scale, 0),
                                         QtCore.QPointF(5 * scale, 6 * scale)]))
    painter.end()
    pixmap.save(path, "PNG")
    return path.replace("\\", "/")


def apply(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    for r, c in ((role.Window, BG), (role.Base, BG), (role.AlternateBase, PANEL),
                 (role.Button, RAISED), (role.ToolTipBase, RAISED), (role.Highlight, ACCENT)):
        pal.setColor(r, QtGui.QColor(c))
    for r, c in ((role.WindowText, TEXT), (role.Text, TEXT), (role.ButtonText, TEXT),
                 (role.ToolTipText, TEXT), (role.HighlightedText, ACCENT_TEXT),
                 (role.PlaceholderText, MUTED)):
        pal.setColor(r, QtGui.QColor(c))
    app.setPalette(pal)
    sheet = QSS
    try:
        sheet = (sheet.replace("@ARROW@", _arrow_image(MUTED, "arrow_down.png"))
                      .replace("@ARROW_OFF@", _arrow_image("#4a535d", "arrow_down_off.png")))
    except OSError:
        pass                        # no arrows is better than no window
    app.setStyleSheet(sheet)
