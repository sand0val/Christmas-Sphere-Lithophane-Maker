"""Fonts for the lettering: which ones this computer has, and how to load them.

Fonts are known by the name Windows lists them under ("Cooper Black", "Segoe
Script Bold").  On Windows the list comes straight from the registry, which
takes a few milliseconds; opening every font file to ask for its name takes
seconds.  Elsewhere the usual font folders are searched.  If a project names
a font this computer does not have, the lettering falls back to the default
and the checklist says so.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from PIL import ImageFont

# Fonts that suit an ornament, sturdy ones first: a printed letter is only as
# good as its thinnest stroke.  (Measured with overlay.stroke_share: the typical
# stroke of all but the last three is at least an eighth of the height of its
# capitals; Curlz MT, Gabriola and Ink Free are finer and want Thicken letters.)
FESTIVE = (
    "Cooper Black", "Harlow Solid Italic", "Forte", "Script MT Bold", "Brush Script MT Italic",
    "Segoe Script Bold", "Segoe Print Bold", "Comic Sans MS Bold", "Arial Rounded MT Bold",
    "Kristen ITC", "Magneto Bold", "Lucida Handwriting Italic", "Lucida Calligraphy Italic",
    "Monotype Corsiva", "Old English Text MT", "Georgia Bold Italic", "Georgia Bold",
    "Snap ITC", "Ravie", "Jokerman", "Showcard Gothic", "Bauhaus 93", "Broadway",
    "Rockwell Extra Bold", "Gill Sans Ultra Bold", "Elephant", "Algerian", "Impact",
    "Arial Black", "Curlz MT", "Gabriola", "Ink Free",
)
# First of these that is installed is the font new lettering starts with.
DEFAULTS = ("Cooper Black", "Georgia Bold Italic", "Segoe Print Bold", "Comic Sans MS Bold",
            "Arial Bold", "DejaVu Sans Bold")
BUILT_IN = "Built-in"           # Pillow's own font: always there

# Picture and icon fonts: no letters in them.  (Segoe UI Symbol is not one of them: it has
# the alphabet as well as hearts, stars and snowflakes.)
_NOT_LETTERS = ("wingdings", "webdings", "bookshelf symbol", "marlett", "mdl2", "fluent icons",
                "mt extra", "ms outlook", "ms reference specialty", "emoji", "segoe ui historic",
                "holo", "awesome", "icons", "codicon", "remixicon", "phosphor", "webfont", "dingbat")
_NOT_LETTERS_EXACT = ("symbol",)
_SUFFIXES = (" (TrueType)", " (OpenType)", " (Type 1)")


def _windows_fonts() -> dict[str, tuple[str, int]]:
    import winreg

    found: dict[str, tuple[str, int]] = {}
    key_name = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
    system_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    user_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")
    for hive, folder in ((winreg.HKEY_LOCAL_MACHINE, system_dir), (winreg.HKEY_CURRENT_USER, user_dir)):
        try:
            key = winreg.OpenKey(hive, key_name)
        except OSError:
            continue
        with key:
            index = 0
            while True:
                try:
                    name, value, _kind = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                if not isinstance(value, str) or not value.lower().endswith((".ttf", ".otf", ".ttc")):
                    continue                        # old bitmap fonts (.fon) cannot be scaled
                path = value if os.path.isabs(value) else os.path.join(folder, value)
                for suffix in _SUFFIXES:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                # a collection file holds several fonts: "Cambria & Cambria Math"
                for face, part in enumerate(name.split(" & ")):
                    found.setdefault(part.strip(), (path, face))
    return found


def _folder_fonts() -> dict[str, tuple[str, int]]:
    """Fonts found by looking into the usual folders (slower: every file is opened)."""
    home = os.path.expanduser("~")
    folders = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.join(home, ".fonts"),
               os.path.join(home, ".local", "share", "fonts"), "/Library/Fonts",
               "/System/Library/Fonts", os.path.join(home, "Library", "Fonts")]
    found: dict[str, tuple[str, int]] = {}
    for folder in folders:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if not name.lower().endswith((".ttf", ".otf", ".ttc")) or len(found) > 600:
                    continue
                path = os.path.join(root, name)
                for face in range(8 if name.lower().endswith(".ttc") else 1):
                    try:
                        family, style = ImageFont.truetype(path, 20, index=face).getname()
                    except Exception:
                        break
                    label = family if style in (None, "", "Regular") else f"{family} {style}"
                    found.setdefault(label, (path, face))
    return found


@lru_cache(maxsize=1)
def installed() -> dict[str, tuple[str, int]]:
    """name -> (file, face number) of every lettering font on this computer."""
    try:
        found = _windows_fonts() if sys.platform == "win32" else _folder_fonts()
    except Exception:           # an unreadable registry or folder must not stop the app
        found = {}
    return {name: where for name, where in sorted(found.items(), key=lambda item: item[0].lower())
            if os.path.isfile(where[0]) and name.lower() not in _NOT_LETTERS_EXACT
            and not any(word in name.lower() for word in _NOT_LETTERS)}


def festive() -> list[str]:
    """The fonts of FESTIVE that this computer has, in that order."""
    have = installed()
    return [name for name in FESTIVE if name in have]


def default_font() -> str:
    have = installed()
    for name in DEFAULTS + FESTIVE:
        if name in have:
            return name
    return BUILT_IN


def has(name: str) -> bool:
    return name == BUILT_IN or name in installed()


@lru_cache(maxsize=64)
def _load(path: str, face: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=face)


@lru_cache(maxsize=None)
def _face(path: str, name: str, listed: int) -> int:
    """Which font inside a collection file (.ttc) is the one called `name`.

    Windows lists the fonts of a collection in one registry entry, and not
    always in the order they have in the file ("Yu Gothic UI Bold" and "Yu
    Gothic UI Semibold" are the other way round), so the file itself is asked."""
    if not path.lower().endswith(".ttc"):
        return listed
    plain = None
    for face in range(24):
        try:
            family, style = ImageFont.truetype(path, 20, index=face).getname()
        except Exception:
            break
        if name == f"{family} {style}":
            return face
        if name == family and plain is None and style in (None, "", "Regular"):
            plain = face
    return listed if plain is None else plain


def _where(name: str) -> tuple[str, int] | None:
    """(file, face number) of an installed font, or None."""
    listed = installed().get(name)
    if listed is None:
        return None
    try:
        return listed[0], _face(listed[0], name, listed[1])
    except Exception:
        return listed


def load(name: str, size: int):
    """The named font at `size` pixels, or the default one if it is missing or
    cannot be read.  Never raises."""
    size = max(4, int(size))
    for candidate in (name, default_font()):
        where = _where(candidate)
        if where is None:
            continue
        try:
            return _load(where[0], where[1], size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:               # very old Pillow: one fixed size only
        return ImageFont.load_default()


def resolve(name: str) -> str:
    """The font that load(name, ...) really gives: `name` itself when this
    computer has it and it can be read, otherwise the stand-in."""
    for candidate in (name, default_font()):
        where = _where(candidate)
        if where is None:
            continue
        try:
            _load(where[0], where[1], 24)
            return candidate
        except Exception:
            continue
    return BUILT_IN
