#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_cli.py - headless slicing with the Bambu Studio CLI (Windows).

    from slice_cli import slice_with_bambu
    r = slice_with_bambu("ornament.stl", "ornament.gcode.3mf",
                         {"layer_height": 0.12, "wall_loops": 3, "sparse_infill_density": "15%",
                          "brim_type": "outer_only", "brim_width": 5, "enable_support": False,
                          "seam_position": "back", "wall_generator": "arachne",
                          "outer_wall_speed": 60, "inner_wall_speed": 120})
    if r["ok"]:
        print(r["print_time_s"], r["filament_g"], r["layer_count"], r["support_used"])
    else:
        print(r["error"], r["log_tail"])

Only the Python standard library is needed.  Developed and tested against Bambu Studio 02.07.01.62
(system profiles 02.07.00.08) for "Bambu Lab A1 0.4 nozzle" / "0.12mm Fine @BBL A1" /
"Bambu PLA Basic @BBL A1".

The command that is run (see result["command"]):

    bambu-studio.exe --debug 3 --orient 0 --arrange 1 --ensure-on-bed
        --curr-bed-type "Textured PEI Plate"
        --load-settings "<work>\\machine.json;<work>\\process.json"
        --load-filaments "<work>\\filament.json"
        --slice 0 --outputdir "<work>\\out" --export-3mf <name>.gcode.3mf  <model.stl>

Things this module does because the CLI needs them (all found by experiment, see the report):

 1. FLATTENED PRESETS.  The CLI does NOT resolve "inherits" (nor the machine "include" templates).
    Passing a system preset file straight from resources\\profiles\\BBL still exits 0 with
    "Success." but silently slices with built-in defaults (0.2 mm layers, 200x200 bed, generic
    start g-code, filament density 0).  So the inheritance chain is merged here and complete
    machine/process/filament JSON files are written to a work directory.
 2. OVERRIDES are written into those JSON files.  A preset that was modified is emitted as a
    "User" preset that inherits the system preset (plus different_settings_to_system), so the
    3mf describes itself honestly.  Every override is read back from the effective settings
    embedded in the output (Metadata/project_settings.config) and reported; the CLI silently
    replaces invalid enum values with built-in defaults and still exits 0.
 3. CONSOLE-LESS PARENT.  bambu-studio.exe is a GUI-subsystem launcher that calls
    AttachConsole(ATTACH_PARENT_PROCESS) and re-opens stdout/stderr on CONOUT$, which defeats
    redirection.  The CLI is therefore started from a small DETACHED helper process (this same
    file run with --_worker); with no parent console the redirection survives and the log can
    be captured.
 4. DIALOG WATCHDOG.  If a blacklisted DLL (e.g. Nahimic "NahimicOSD.dll") is injected into the
    process, Bambu Studio shows a modal "Warning" message box before doing anything, which
    hangs a headless run forever.  The helper finds that specific box on the process it started
    and presses OK.  Any other dialog/visible window is treated as a blocker: the process is
    killed and the dialog text is returned.  Only the process started here is ever killed.
 5. RESULT CHECKS.  Exit code 0 is not sufficient ("--slice 2" on a one-plate STL exits 0 and
    writes a 3mf with no g-code), so the output is opened and checked for a g-code plate.
 6. SHORT SCRATCH PATH.  The CLI has no long-path support: an --outputdir longer than 227
    characters fails with exit -100 ("Cannot open file for writing").  The scratch directory
    therefore defaults to the temp dir and the finished 3mf is moved to out_3mf_path by Python
    (spaces and non-ASCII characters in any path are fine).
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["slice_with_bambu", "flatten_preset", "CLI_EXIT_CODES"]

DEFAULT_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
DEFAULT_MACHINE = "Bambu Lab A1 0.4 nozzle"
DEFAULT_PROCESS = "0.12mm Fine @BBL A1"          # also present: "0.08mm Extra Fine @BBL A1", "0.08mm High Quality @BBL A1"
DEFAULT_FILAMENT = "Bambu PLA Basic @BBL A1"
DEFAULT_BED_TYPE = "Textured PEI Plate"            # A1 default; others: "Cool Plate", "Engineering Plate",
                                                   # "High Temp Plate", "Supertack Plate"

# Return codes of the CLI.  The message order was read from BambuStudio.dll 02.07.01.62; codes
# marked (seen) were actually observed, the rest follow from the position in that table.
CLI_EXIT_CODES: Dict[int, str] = {
    0: "Success. (seen)",
    -1: "Failed setting up server environment.",
    -2: "Invalid parameters to the slicer. (seen: unknown option / bad option syntax)",
    -3: "The input files to the slicer are not found. (seen: missing model or preset file)",
    -4: "File list order to the slicer is invalid. Please make sure the 3mf in the first place.",
    -5: "The input preset file is invalid and can not be parsed. (seen: broken JSON)",
    -6: "The input model file to the slicer can not be parsed.",
    -7: "Unsupported printer technology (not FDM).",
    -8: "Unsupported CLI instruction.",
    -9: "Failed copying objects.",
    -10: "Failed scaling an object to fit the plate.",
    -11: "Failed exporting STL files.",
    -12: "Failed exporting OBJ files.",
    -13: "Failed exporting 3mf files. (seen: absolute --export-3mf path together with --outputdir)",
    -14: "Out of memory during slicing.",
    -15: "The selected printer is not supported.",
    -16: "The selected printer is not compatible with the 3mf.",
    -17: "The selected printer is not compatible with the process preset. (seen)",
    -18: "Invalid parameter value(s) included in the 3mf file.",
    -19: "post_process is not supported under CLI.",
    -20: "The selected printer's bed size is smaller than the bed size used in the print profile.",
    -21: "An error occurred when auto-arranging object(s).",
    -22: "An error occurred when auto-orienting object(s).",
    -23: "You cannot change the Printable Area, Printable Height, and Exclude Area in Printer Settings.",
    -24: "Unsupported 3MF version.",
    -25: "Unsupported features were found in this 3MF file.",
    -50: "One of the plates is empty or has no object fully inside it. (seen: model larger than the bed, or --arrange 0)",
    -51: "There are some incorrect slicing parameters. (seen: e.g. layer height > nozzle diameter; reason is on stderr)",
    -52: "Some objects are located over the boundary of the heated bed.",
    -53: "Failed creating directory when exporting cache data.",
    -54: "Failed exporting cache data.",
    -55: "Cache data not found.",
    -56: "Cache data can not be parsed.",
    -57: "Failed importing cache data.",
    -58: "Slicing time of a certain plate exceeds the limit.",
    -59: "Triangle count of single plate exceeds the limit.",
    -60: "No printable objects to slice after skipping.",
    -61: "Filaments are not compatible with the plate type. (seen: bed type whose temperature is 0 for the filament)",
    -62: "The temperature difference of the filaments used is too large.",
    -63: "Object conflicts were detected when using print-by-object mode.",
    -64: "Object conflicts were detected.",
    -65: "Some slicing parameters cannot work with Spiral Vase mode.",
    -66: "Some filaments cannot be mapped to correct extruders for multi-extruder Printer.",
    -67: "Not support printing 2 or more TPU filaments.",
    -68: "Some filaments cannot be printed on the extruder mapped to.",
    -100: "Failed slicing the model.",
    -101: "G-code conflicts detected after slicing.",
    -102: "Found G-code in unprintable area of multi-extruder printers after slicing.",
    -103: "Found some filament unprintable at first layer on current Plate.",
    -104: "Found G-code outside of the printable area.",
    -105: "Found G-code in the wrapping detect area.",
}

_KINDS = ("machine", "process", "filament")
_INDEX_KEYS = {"machine": "machine_list", "process": "process_list", "filament": "filament_list"}
# keys of a preset JSON that are bookkeeping, not settings
_META_KEYS = {"type", "name", "inherits", "from", "setting_id", "instantiation", "include", "description",
              "filament_id", "version", "different_settings_to_system"}
_INJECTED_DLL_MARKER = "have been injected into the BambuStudio process"

_IS_WINDOWS = os.name == "nt"
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
# Path limits of the CLI (no long-path support), measured on 02.07.01.62:
_MAX_OUTPUTDIR_LEN = 227     # 228+ fails: "<outputdir>\plate_1.gcode.tmp.postprocessed" reaches MAX_PATH
_MAX_PATH_LEN = 255          # preset files work up to 255 characters, the model up to 259


# --------------------------------------------------------------------------------------------
# preset handling
# --------------------------------------------------------------------------------------------
def _version_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text or "")) or (0,)


class _Profiles:
    """Locates system preset files of one vendor (BBL) and flattens them."""

    def __init__(self, root: str, vendor: str = "BBL"):
        self.root = root
        self.dir = os.path.join(root, vendor)
        self.version = ""
        self._index: Dict[Tuple[str, str], str] = {}
        self._scanned: Dict[str, Dict[str, str]] = {}
        idx_path = os.path.join(root, vendor + ".json")
        if os.path.isfile(idx_path):
            with open(idx_path, encoding="utf-8") as fh:
                idx = json.load(fh)
            self.version = str(idx.get("version", ""))
            for kind, key in _INDEX_KEYS.items():
                for entry in idx.get(key, []) or []:
                    if "name" in entry and "sub_path" in entry:
                        self._index[(kind, entry["name"])] = os.path.join(self.dir, *entry["sub_path"].split("/"))

    def path(self, kind: str, name: str) -> str:
        cand = self._index.get((kind, name))
        if cand and os.path.isfile(cand):
            return cand
        cand = os.path.join(self.dir, kind, name + ".json")
        if os.path.isfile(cand):
            return cand
        # last resort: look for a file in <kind>/ whose "name" field matches
        if kind not in self._scanned:
            table: Dict[str, str] = {}
            folder = os.path.join(self.dir, kind)
            if os.path.isdir(folder):
                for fn in os.listdir(folder):
                    if fn.lower().endswith(".json"):
                        full = os.path.join(folder, fn)
                        try:
                            with open(full, encoding="utf-8") as fh:
                                table[str(json.load(fh).get("name"))] = full
                        except (OSError, ValueError):
                            pass
            self._scanned[kind] = table
        cand = self._scanned[kind].get(name)
        if cand:
            return cand
        raise FileNotFoundError("%s preset %r not found under %s" % (kind, name, self.dir))

    def flatten(self, kind: str, name: str, _stack: Tuple[str, ...] = ()) -> Dict[str, Any]:
        """Merge a preset with everything it inherits/includes.  Order: parent chain, then the
        preset's "include" templates, then the preset's own keys (own keys win)."""
        if name in _stack:
            raise ValueError("preset inheritance cycle: %s" % " -> ".join(_stack + (name,)))
        with open(self.path(kind, name), encoding="utf-8") as fh:
            data = json.load(fh)
        merged: Dict[str, Any] = {}
        parent = data.get("inherits")
        if parent:
            merged.update(self.flatten(kind, parent, _stack + (name,)))
        for inc in data.get("include") or []:
            merged.update(self.flatten(kind, inc, _stack + (name,)))
        merged.update(data)
        return merged


def _find_profiles(exe: str, profiles_root: Optional[str]) -> _Profiles:
    """Pick the system profile tree: the installer's copy, unless the copy the GUI maintains in
    %APPDATA%\\BambuStudio\\system is strictly newer.  Both are only read."""
    if profiles_root:
        return _Profiles(profiles_root)
    cands = [os.path.join(os.path.dirname(exe), "resources", "profiles")]
    appdata = os.environ.get("APPDATA")
    if appdata:
        cands.append(os.path.join(appdata, "BambuStudio", "system"))
    best: Optional[_Profiles] = None
    for root in cands:
        if os.path.isdir(os.path.join(root, "BBL")):
            prof = _Profiles(root)
            if best is None or _version_tuple(prof.version) > _version_tuple(best.version):
                best = prof
    if best is None:
        raise FileNotFoundError("no BBL system profiles found in: %s" % "; ".join(cands))
    return best


def flatten_preset(kind: str, name: str, exe: str = DEFAULT_EXE, profiles_root: Optional[str] = None) -> Dict[str, Any]:
    """Public helper: the fully resolved settings of a BBL system preset.
    kind is "machine", "process" or "filament"."""
    return _find_profiles(exe, profiles_root).flatten(kind, name)


def _scalar_str(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text
    return str(value)


def _to_cfg_value(value: Any, base: Any) -> Any:
    """Convert a Python override into the string / list-of-strings form used by preset JSON,
    matching the shape (scalar vs per-extruder list) of the value it replaces."""
    if isinstance(value, (list, tuple)):
        vals = [_scalar_str(v) for v in value]
        if isinstance(base, list) or base is None:
            return vals
        if len(vals) == 1:
            return vals[0]
        raise ValueError("a list was given for a scalar setting")
    text = _scalar_str(value)
    if isinstance(base, list):
        return [text] * max(1, len(base))
    return text


def _norm_scalar(value: Any) -> Tuple[str, Any]:
    text = str(value).strip()
    low = text.lower()
    if low in ("true", "false"):
        return ("num", 1.0 if low == "true" else 0.0)
    core = text[:-1] if text.endswith("%") else text
    try:
        return ("num", round(float(core), 9))
    except ValueError:
        return ("str", text.replace(",", "x") if re.fullmatch(r"[\d.]+[x,][\d.]+", text) else text)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare a requested value with the effective one ("15" == "15%", "0.20" == "0.2",
    scalar == one-element list, n identical elements == one element)."""
    la = [_norm_scalar(x) for x in (a if isinstance(a, list) else [a])]
    lb = [_norm_scalar(x) for x in (b if isinstance(b, list) else [b])]
    if len(la) == len(lb):
        return la == lb
    return len(set(la)) == 1 and len(set(lb)) == 1 and la[0] == lb[0]


def _same_or_truncated(passed: Any, effective: Any) -> bool:
    """Equal, or the CLI merely shortened a list to the number of filaments/extruders in use."""
    if _values_equal(passed, effective):
        return True
    if isinstance(passed, list) and isinstance(effective, list) and 0 < len(effective) < len(passed):
        return _values_equal(passed[:len(effective)], effective)
    return False


def _build_configs(prof: _Profiles, names: Dict[str, str], overrides: Dict[str, Any],
                   custom_suffix: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Returns (configs, requested, final_names).  requested[kind][key] is the value written."""
    base = {kind: prof.flatten(kind, names[kind]) for kind in _KINDS}
    requested: Dict[str, Dict[str, Any]] = {kind: {} for kind in _KINDS}
    for raw_key, value in (overrides or {}).items():
        key = str(raw_key)
        kind = None
        if ":" in key and key.split(":", 1)[0] in _KINDS:
            kind, key = key.split(":", 1)
        if key in _META_KEYS:
            raise ValueError("%r is preset bookkeeping, not a setting" % key)
        if kind is None:
            # process first: that is where layer height, walls, infill, brim, support, seam, speeds live
            kind = next((k for k in ("process", "filament", "machine") if key in base[k]), "process")
        try:
            requested[kind][key] = _to_cfg_value(value, base[kind].get(key))
        except ValueError as exc:
            raise ValueError("override %r: %s" % (raw_key, exc))
    configs: Dict[str, Dict[str, Any]] = {}
    final_names: Dict[str, str] = {}
    for kind in _KINDS:
        cfg = dict(base[kind])
        for drop in ("include", "description"):
            cfg.pop(drop, None)
        changed = sorted(k for k, v in requested[kind].items() if k not in base[kind] or not _values_equal(v, base[kind][k]))
        cfg.update(requested[kind])
        cfg["type"] = kind
        cfg["instantiation"] = "true"
        if changed:
            # a modified preset: a "User" preset inheriting the system one
            final_names[kind] = names[kind] + custom_suffix
            cfg["name"] = final_names[kind]
            cfg["from"] = "User"
            cfg["inherits"] = names[kind]
            cfg.pop("setting_id", None)
            diff = ";".join(changed)
            # the CLI reads element 0 from the process and filament files and the last element
            # from the machine file (observed); the same string everywhere satisfies all three
            cfg["different_settings_to_system"] = [diff, diff, diff]
        else:
            final_names[kind] = names[kind]
            cfg["name"] = names[kind]
            cfg["from"] = "system"
            cfg["inherits"] = ""
        configs[kind] = cfg
    return configs, requested, final_names


# --------------------------------------------------------------------------------------------
# running the CLI (used by the detached worker, or directly as a fallback)
# --------------------------------------------------------------------------------------------
def _signed32(code: Optional[int]) -> Optional[int]:
    if code is None:
        return None
    code &= 0xFFFFFFFF
    return code - 0x100000000 if code & 0x80000000 else code


class _WinWatch:
    """Enumerates the top-level windows of one process and can press a dialog button."""

    WM_CLOSE = 0x0010
    WM_COMMAND = 0x0111
    SYNCHRONIZE = 0x00100000

    def __init__(self) -> None:
        import ctypes.wintypes as wt
        self.wt = wt
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.ENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        u = self.user32
        u.EnumWindows.argtypes = [self.ENUMPROC, wt.LPARAM]
        u.EnumChildWindows.argtypes = [wt.HWND, self.ENUMPROC, wt.LPARAM]
        u.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        u.GetWindowThreadProcessId.restype = wt.DWORD
        u.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u.IsWindowVisible.argtypes = [wt.HWND]
        u.GetDlgCtrlID.argtypes = [wt.HWND]
        u.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
        k = self.kernel32
        k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        k.OpenProcess.restype = wt.HANDLE
        k.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        k.WaitForSingleObject.restype = wt.DWORD
        k.CloseHandle.argtypes = [wt.HANDLE]

    def _text(self, hwnd: Any, cls: bool = False, size: int = 4096) -> str:
        buf = ctypes.create_unicode_buffer(size)
        (self.user32.GetClassNameW if cls else self.user32.GetWindowTextW)(hwnd, buf, size)
        return buf.value

    def windows(self, pid: int) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []

        def cb(hwnd: Any, _lparam: Any) -> bool:
            wpid = self.wt.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value == pid:
                found.append({"hwnd": hwnd, "visible": bool(self.user32.IsWindowVisible(hwnd)),
                              "class": self._text(hwnd, cls=True, size=256), "title": self._text(hwnd, size=512)})
            return True

        self.user32.EnumWindows(self.ENUMPROC(cb), 0)
        return found

    def children(self, hwnd: Any) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []

        def cb(child: Any, _lparam: Any) -> bool:
            found.append({"hwnd": child, "class": self._text(child, cls=True, size=256),
                          "text": self._text(child), "id": self.user32.GetDlgCtrlID(child)})
            return True

        self.user32.EnumChildWindows(hwnd, self.ENUMPROC(cb), 0)
        return found

    def press(self, dialog_hwnd: Any, button: Dict[str, Any]) -> bool:
        # NB: the single OK button of an MB_OK message box has control id 2 (IDCANCEL), not 1,
        # so the id is taken from the button itself.
        return bool(self.user32.PostMessageW(dialog_hwnd, self.WM_COMMAND, button["id"], button["hwnd"] or 0))

    def close(self, hwnd: Any) -> bool:
        return bool(self.user32.PostMessageW(hwnd, self.WM_CLOSE, 0, 0))

    def open_process(self, pid: int) -> Any:
        return self.kernel32.OpenProcess(self.SYNCHRONIZE, False, pid)

    def has_exited(self, handle: Any) -> bool:
        return bool(handle) and self.kernel32.WaitForSingleObject(handle, 0) == 0


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    """Kill the process WE started (and anything it spawned).  Never kills by image name."""
    try:
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)
    except OSError:
        pass
    try:
        if proc.poll() is None:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(15)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _run_cli(cmd: List[str], cwd: str, stdout_path: str, stderr_path: str, timeout_s: float,
             parent_pid: Optional[int] = None, dialog_grace_s: float = 6.0) -> Dict[str, Any]:
    """Run the CLI to completion.  Presses OK on the injected-DLL warning, kills the process on
    timeout / unexpected dialog / parent death.  Returns a status dict."""
    status: Dict[str, Any] = {"pid": None, "exit_code": None, "timed_out": False, "killed": None,
                              "dialogs": [], "elapsed_s": None}
    watch = _WinWatch() if _IS_WINDOWS else None
    parent_handle = watch.open_process(parent_pid) if (watch and parent_pid) else None
    t0 = time.time()
    with open(stdout_path, "wb") as fo, open(stderr_path, "wb") as fe:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)
        status["pid"] = proc.pid
        pressed: Dict[Any, List[float]] = {}     # hwnd -> times we pressed OK
        suspects: Dict[Any, float] = {}          # hwnd -> first time an unexpected window was seen
        try:
            while True:
                if proc.poll() is not None:
                    break
                now = time.time()
                if now - t0 > timeout_s:
                    status["timed_out"] = True
                    status["killed"] = "timeout after %.0f s" % timeout_s
                    _kill_tree(proc)
                    break
                if watch and parent_handle and watch.has_exited(parent_handle):
                    status["killed"] = "calling process exited"
                    _kill_tree(proc)
                    break
                if watch:
                    blocker = None
                    for win in watch.windows(proc.pid):
                        is_dialog = win["class"] == "#32770"
                        if not (is_dialog or (win["visible"] and win["title"])):
                            continue
                        kids = watch.children(win["hwnd"]) if is_dialog else []
                        text = "\n".join(k["text"] for k in kids if k["class"] == "Static" and k["text"])
                        if is_dialog and _INJECTED_DLL_MARKER in text:
                            presses = pressed.setdefault(win["hwnd"], [])
                            if not presses or now - presses[-1] > 1.0:
                                buttons = [k for k in kids if k["class"] == "Button"]
                                if len(presses) < 3 and buttons:
                                    watch.press(win["hwnd"], buttons[0])
                                else:
                                    watch.close(win["hwnd"])
                                if not presses:
                                    status["dialogs"].append({"title": win["title"], "text": text,
                                                              "action": "dismissed (OK pressed)"})
                                presses.append(now)
                            continue
                        first = suspects.setdefault(win["hwnd"], now)
                        if now - first > dialog_grace_s:
                            blocker = {"title": win["title"], "class": win["class"], "text": text,
                                       "action": "unexpected window - process killed"}
                    if blocker:
                        status["dialogs"].append(blocker)
                        status["killed"] = "blocked by unexpected window %r" % blocker["title"]
                        _kill_tree(proc)
                        break
                # the warning box appears ~0.5 s after start: poll fast at first so it only flashes
                time.sleep(0.03 if now - t0 < 15 else 0.2)
        except BaseException:
            _kill_tree(proc)
            raise
        finally:
            if watch and parent_handle:
                watch.kernel32.CloseHandle(parent_handle)
        status["exit_code"] = _signed32(proc.poll())
    status["elapsed_s"] = round(time.time() - t0, 2)
    return status


def _worker_main(job_path: str) -> int:
    """Entry point of the detached helper process (python slice_cli.py --_worker job.json)."""
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    try:
        status = _run_cli(job["cmd"], job["cwd"], job["stdout"], job["stderr"], float(job["timeout_s"]),
                          parent_pid=job.get("parent_pid"))
    except Exception as exc:  # report instead of dying silently
        status = {"pid": None, "exit_code": None, "timed_out": False, "killed": None, "dialogs": [],
                  "elapsed_s": None, "worker_error": "%s: %s" % (type(exc).__name__, exc)}
    tmp = job["status"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh)
    os.replace(tmp, job["status"])
    return 0


def _run_detached(job: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
    """Run the CLI through a console-less helper so that its stdout/stderr can be captured."""
    job_path = os.path.join(work_dir, "job.json")
    with open(job_path, "w", encoding="utf-8") as fh:
        json.dump(job, fh)
    worker_log = os.path.join(work_dir, "worker.log")
    with open(worker_log, "wb") as wl:
        helper = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_worker", job_path],
                                  cwd=work_dir, stdin=subprocess.DEVNULL, stdout=wl, stderr=subprocess.STDOUT,
                                  creationflags=_DETACHED_PROCESS)
        try:
            helper.wait(float(job["timeout_s"]) + 90)
        except BaseException:       # timeout, KeyboardInterrupt, ...: never leave the slicer running
            _kill_tree(helper)
            raise
    if os.path.isfile(job["status"]):
        with open(job["status"], encoding="utf-8") as fh:
            return json.load(fh)
    detail = ""
    try:
        with open(worker_log, encoding="utf-8", errors="replace") as fh:
            detail = fh.read()[-2000:]
    except OSError:
        pass
    return {"pid": None, "exit_code": None, "timed_out": False, "killed": None, "dialogs": [], "elapsed_s": None,
            "worker_error": "helper exited with %s without a status file. %s" % (helper.returncode, detail)}


# --------------------------------------------------------------------------------------------
# reading results
# --------------------------------------------------------------------------------------------
def _parse_duration(text: str) -> Optional[int]:
    """'1d 2h 3m 4s' -> seconds."""
    total, hit = 0, False
    for num, unit in re.findall(r"(\d+)\s*([dhms])", text):
        total += int(num) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        hit = True
    return total if hit else None


def _sum_floats(text: str) -> Optional[float]:
    vals = re.findall(r"-?\d+(?:\.\d+)?", text)
    return round(sum(float(v) for v in vals), 4) if vals else None


def _read_gcode(zf: zipfile.ZipFile, name: str, scan_features: bool) -> Dict[str, Any]:
    """Header values of a plate g-code and (optionally) how often each '; FEATURE:' occurs."""
    info: Dict[str, Any] = {"features": {}}
    feats: "collections.Counter[str]" = collections.Counter()
    in_header = True
    with zf.open(name) as fh:
        for n, raw in enumerate(fh):
            if in_header:
                line = raw.decode("utf-8", "replace").strip()
                m = re.match(r"; BambuStudio (\S+)", line)
                if m:
                    info["slicer_version"] = m.group(1)
                m = re.match(r"; model printing time: (.*?); total estimated time: (.*)", line)
                if m:
                    info["model_print_time_s"] = _parse_duration(m.group(1))
                    info["total_time_s"] = _parse_duration(m.group(2))
                    info["time_text"] = m.group(2).strip()
                m = re.match(r"; total layer number: (\d+)", line)
                if m:
                    info["layer_count"] = int(m.group(1))
                m = re.match(r"; total filament length \[mm\] : (.*)", line)
                if m:
                    info["filament_mm"] = _sum_floats(m.group(1))
                m = re.match(r"; total filament weight \[g\] : (.*)", line)
                if m:
                    info["filament_g"] = _sum_floats(m.group(1))
                m = re.match(r"; max_z_height: (.*)", line)
                if m:
                    info["max_z_mm"] = _sum_floats(m.group(1))
                if line.startswith("; HEADER_BLOCK_END") or n > 200:
                    in_header = False
                    if not scan_features:
                        break
            elif raw.startswith(b"; FEATURE: "):
                feats[raw[11:].strip().decode("utf-8", "replace")] += 1
    info["features"] = dict(feats)
    return info


def _read_3mf(path: str, scan_features: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"gcode_files": [], "settings": None, "plate": {}, "filaments": [], "gcode": {},
                           "slice_warnings": [], "client_version": None}
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        out["gcode_files"] = sorted(n for n in names if re.fullmatch(r"Metadata/plate_\d+\.gcode", n))
        if "Metadata/project_settings.config" in names:
            out["settings"] = json.loads(zf.read("Metadata/project_settings.config").decode("utf-8"))
        if "Metadata/slice_info.config" in names:
            root = ET.fromstring(zf.read("Metadata/slice_info.config"))
            for item in root.iter("header_item"):
                if item.get("key") == "X-BBL-Client-Version":
                    out["client_version"] = item.get("value")
            plate = root.find("plate")
            if plate is not None:
                out["plate"] = {m.get("key"): m.get("value") for m in plate.findall("metadata")}
                out["filaments"] = [dict(f.attrib) for f in plate.findall("filament")]
                out["slice_warnings"] = [dict(w.attrib) for w in plate.findall("warning")]
                ranges = [r.get("layer_ranges", "") for r in plate.iter("layer_filament_list")]
                tops = [int(r.split()[-1]) for r in ranges if r.split()]
                if tops:
                    out["plate"]["_layer_count"] = max(tops) + 1
        if out["gcode_files"]:
            out["gcode"] = _read_gcode(zf, out["gcode_files"][0], scan_features)
    return out


_WORK_FILES = ("machine.json", "process.json", "filament.json", "job.json", "status.json", "status.json.tmp",
               "worker.log", "cli_stdout.log", "cli_stderr.log")
_DIAGNOSTIC_FILES = ("machine.json", "process.json", "filament.json", "cli_stdout.log", "cli_stderr.log", "worker.log")


def _clean_work_dir(work_dir: str, remove_dir: bool, diagnostics_too: bool = True) -> None:
    """Remove only what this module (or the CLI on its behalf) put into the work directory:
    the files listed above, the "out" sub-directory and backup folders the CLI creates in its
    working directory ("HH_MM_SS#pid#n").  With diagnostics_too=False the presets, the logs and
    out\\result.json are kept, the bulky g-code / 3mf are not.  Nothing else is ever deleted."""
    out_dir = os.path.join(work_dir, "out")
    if diagnostics_too:
        shutil.rmtree(out_dir, ignore_errors=True)
    elif os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if name != "result.json":
                full = os.path.join(out_dir, name)
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    try:
                        os.remove(full)
                    except OSError:
                        pass
    try:
        entries = os.listdir(work_dir)
    except OSError:
        return
    for name in entries:
        full = os.path.join(work_dir, name)
        if re.fullmatch(r"\d\d_\d\d_\d\d#\d+#\d+", name) and os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)
        elif name in _WORK_FILES and (diagnostics_too or name not in _DIAGNOSTIC_FILES):
            try:
                os.remove(full)
            except OSError:
                pass
    if remove_dir:
        try:
            os.rmdir(work_dir)          # only succeeds when empty
        except OSError:
            pass


def _short_path(path: str) -> str:
    """8.3 alias of an existing path (shorter and ASCII), or the path itself if unavailable."""
    try:
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
        return buf.value if 0 < n < len(buf) else path
    except (OSError, AttributeError):
        return path


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            keep = [ln.rstrip() for ln in fh if ln.strip()]
    except OSError:
        return ""
    return "\n".join(keep[-lines:])


def _grep(path: str, pattern: str, limit: int = 40) -> List[str]:
    hits: List[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if re.search(pattern, ln):
                    ln = ln.strip()
                    if ln not in hits:
                        hits.append(ln)
    except OSError:
        pass
    return hits[:limit]


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def slice_with_bambu(stl_path: str, out_3mf_path: str, overrides: Optional[Dict[str, Any]] = None, *,
                     process: str = DEFAULT_PROCESS, machine: str = DEFAULT_MACHINE,
                     filament: str = DEFAULT_FILAMENT, bed_type: Optional[str] = DEFAULT_BED_TYPE,
                     timeout_s: float = 1800.0, strict_overrides: bool = True, debug_level: int = 3,
                     extra_args: Optional[List[str]] = None, work_dir: Optional[str] = None,
                     keep_work_dir: Optional[bool] = None, scan_gcode: bool = True, capture_log: bool = True,
                     exe: str = DEFAULT_EXE, profiles_root: Optional[str] = None,
                     custom_suffix: str = " - custom", log_tail_lines: int = 30,
                     arrange: bool = True) -> Dict[str, Any]:
    """Slice one model (STL, or a 3MF - which may carry painted seams) headlessly with Bambu
    Studio and export a sliced .gcode.3mf.

    overrides   {setting: value} applied on top of the presets.  Keys are Bambu setting names
                (layer_height, initial_layer_print_height, wall_loops, sparse_infill_density,
                brim_type, brim_width, enable_support, support_type, seam_position, wall_generator,
                outer_wall_speed, inner_wall_speed, sparse_infill_speed, top_surface_speed, ...).
                A key is routed to the preset that contains it (process first); force one with a
                prefix: "filament:nozzle_temperature", "machine:retraction_length", "process:...".
                Values may be str/int/float/bool (bool -> "1"/"0"); per-extruder list settings
                (all speeds) take a scalar, which is applied to every entry.
    process / machine / filament   names of BBL *system* presets (flattened automatically).
    bed_type    --curr-bed-type; None leaves the CLI default ("Cool Plate").
    arrange     True lets Bambu Studio place the model on the plate - it may also TURN it
                (a half turn was observed).  False keeps the model exactly where the file
                puts it; the model must then already lie on the bed (0..256 mm), otherwise
                the CLI exits with -50.
    timeout_s   wall-clock limit for the slicer process; on expiry only that process is killed.
    strict_overrides   if True, ok is False when an override did not arrive in the output.
    work_dir    scratch directory for the presets, logs and CLI output.  Default: a new
                "bambu_cli_*" folder in the temp dir.  Keep it SHORT - the CLI fails (exit -100)
                when "<work_dir>\\out" is longer than 227 characters.  Only files this module
                creates are ever written to or removed from it.
    keep_work_dir   None: on success remove the scratch files, on failure keep presets + logs
                (result["work_dir"] says where); True: keep everything; False: always remove.
    scan_gcode  count "; FEATURE:" blocks in the g-code (needed for the g-code level support check).
    capture_log run through a console-less helper so the CLI log can be captured.

    Returns a dict:
        ok                  everything below checked out
        exit_code           process exit code as a signed int (0 = success, see CLI_EXIT_CODES)
        error               one-line reason when not ok, else None
        print_time_s        total estimate in seconds, including ~6 min of A1 start-up routine
        model_print_time_s  estimate without the start-up routine
        print_time_text     e.g. "36m 24s"
        filament_g, filament_m, layer_count, layer_height, max_z_mm
        support_used        True if support material was actually GENERATED (g-code features /
                            per-feature times / filament used_for_support)
        support_enabled     the enable_support setting (this is all slice_info's "support_used" means)
        gcode_features      {feature name: number of blocks} when scan_gcode
        warnings            slicer warning names, de-duplicated (from the <warning> elements of
                            Metadata/slice_info.config); warning_details adds level / error_code.
                            Seen with the stock presets: "bed_temperature_too_high_than_filament"
                            (65 C Textured PEI bed vs. PLA's 45 C vitrification temperature) and
                            "not_support_traditional_timelapse" (advisory about timelapse mode)
        overrides_applied / overrides_mismatched      read back from the output file
        cli_adjusted_settings   settings whose effective value differs from the preset that was
                            passed in, e.g. the machine acceleration limits the CLI clamps via
                            resources\\profiles\\BBL\\cli_config.json
        presets             the preset names recorded in the output
        bambu_version, profiles_version, command, out_3mf, work_dir, elapsed_s, timed_out,
        dialogs (message boxes that were handled), stderr, log_errors, log_tail
    """
    t_start = time.time()
    res: Dict[str, Any] = {
        "ok": False, "exit_code": None, "error": None, "print_time_s": None, "model_print_time_s": None,
        "print_time_text": None, "filament_g": None, "filament_m": None, "layer_count": None,
        "layer_height": None, "max_z_mm": None, "support_used": None, "support_enabled": None,
        "gcode_features": {}, "warnings": [], "warning_details": [], "overrides_applied": {},
        "overrides_mismatched": {},
        "cli_adjusted_settings": {}, "presets": {}, "bambu_version": None, "profiles_version": None,
        "command": None, "out_3mf": None, "work_dir": None, "elapsed_s": None, "timed_out": False,
        "dialogs": [], "stderr": "", "log_errors": [], "log_tail": "",
    }

    def finish(error: Optional[str] = None) -> Dict[str, Any]:
        if error and not res["error"]:
            res["error"] = error
        res["ok"] = res["error"] is None
        res["elapsed_s"] = round(time.time() - t_start, 2)
        return res

    stl_path = os.path.abspath(stl_path)
    out_3mf_path = os.path.abspath(out_3mf_path)
    if not _IS_WINDOWS:
        return finish("this module drives the Windows build of Bambu Studio")
    if not os.path.isfile(exe):
        return finish("Bambu Studio not found: %s" % exe)
    if not os.path.isfile(stl_path):
        return finish("model file not found: %s" % stl_path)

    # ---- presets -------------------------------------------------------------------------
    try:
        prof = _find_profiles(exe, profiles_root)
        res["profiles_version"] = prof.version
        configs, requested, final_names = _build_configs(
            prof, {"machine": machine, "process": process, "filament": filament}, overrides or {}, custom_suffix)
    except (OSError, ValueError) as exc:
        return finish("preset problem: %s" % exc)

    # ---- work directory ------------------------------------------------------------------
    # Must be SHORT: the CLI has no long-path support and creates
    # "<outputdir>\plate_1.gcode.tmp.postprocessed", so --outputdir may be at most 227 characters
    # (measured; longer gives exit -100 "Cannot open file for writing").  Default: the temp dir.
    created_work_dir = work_dir is None
    try:
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="bambu_cli_")
        else:
            os.makedirs(work_dir, exist_ok=True)
        work_dir = os.path.abspath(work_dir)
        _clean_work_dir(work_dir, remove_dir=False)          # stale files of an earlier run
        if len(os.path.join(work_dir, "out")) > _MAX_OUTPUTDIR_LEN:
            work_dir = _short_path(work_dir)                 # 8.3 alias, if the volume has them
        out_dir = os.path.join(work_dir, "out")
        if len(out_dir) > _MAX_OUTPUTDIR_LEN:
            if created_work_dir:
                _clean_work_dir(work_dir, remove_dir=True)
            return finish("work directory path is too long for Bambu Studio (%d > %d characters): %s - pass a "
                          "short work_dir" % (len(out_dir), _MAX_OUTPUTDIR_LEN, out_dir))
        if ";" in work_dir:
            return finish("work_dir must not contain ';' (it separates files in --load-settings): %s" % work_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(out_3mf_path) or ".", exist_ok=True)
        res["work_dir"] = work_dir
        paths = {}
        for kind in _KINDS:
            paths[kind] = os.path.join(work_dir, kind + ".json")
            with open(paths[kind], "w", encoding="utf-8") as fh:
                json.dump(configs[kind], fh, indent=4)
    except OSError as exc:
        return finish("cannot prepare the work directory: %s" % exc)
    if len(stl_path) > _MAX_PATH_LEN:
        stl_path = _short_path(stl_path)
        if len(stl_path) > _MAX_PATH_LEN:
            return finish("model path is longer than %d characters, Bambu Studio cannot open it" % _MAX_PATH_LEN)
    stdout_log = os.path.join(work_dir, "cli_stdout.log")
    stderr_log = os.path.join(work_dir, "cli_stderr.log")
    tmp_name = "sliced.gcode.3mf"

    # ---- command -------------------------------------------------------------------------
    cmd = [exe, "--debug", str(int(debug_level)), "--orient", "0", "--arrange", "1" if arrange else "0",
           "--ensure-on-bed"]
    if bed_type:
        cmd += ["--curr-bed-type", bed_type]
    cmd += ["--load-settings", paths["machine"] + ";" + paths["process"],
            "--load-filaments", paths["filament"],
            "--slice", "0", "--outputdir", out_dir, "--export-3mf", tmp_name]
    cmd += list(extra_args or [])
    cmd.append(stl_path)
    res["command"] = subprocess.list2cmdline(cmd)

    # ---- run -----------------------------------------------------------------------------
    job = {"cmd": cmd, "cwd": work_dir, "stdout": stdout_log, "stderr": stderr_log, "timeout_s": timeout_s,
           "status": os.path.join(work_dir, "status.json"), "parent_pid": os.getpid()}
    use_worker = capture_log and not getattr(sys, "frozen", False) and os.path.isfile(os.path.abspath(__file__))
    try:
        if use_worker:
            status = _run_detached(job, work_dir)
        else:
            status = _run_cli(cmd, work_dir, stdout_log, stderr_log, timeout_s)
    except subprocess.TimeoutExpired:
        status = {"exit_code": None, "timed_out": True, "killed": "helper did not return", "dialogs": []}
    res["exit_code"] = status.get("exit_code")
    res["timed_out"] = bool(status.get("timed_out"))
    res["dialogs"] = status.get("dialogs") or []
    try:
        with open(stderr_log, encoding="utf-8", errors="replace") as fh:
            res["stderr"] = fh.read().strip()[-4000:]
    except OSError:
        pass
    res["log_tail"] = _tail(stdout_log, log_tail_lines)
    res["log_errors"] = _grep(stdout_log, r"\[(error|fatal)\]|substituted when loading|old_value =")
    for line in _grep(stdout_log, r"Current BambuStudio Version", 1):
        res["bambu_version"] = line.split("Version")[-1].strip()

    # ---- interpret -----------------------------------------------------------------------
    error: Optional[str] = None
    result_json: Dict[str, Any] = {}
    rj_path = os.path.join(out_dir, "result.json")
    if os.path.isfile(rj_path):
        try:
            with open(rj_path, encoding="utf-8") as fh:
                result_json = json.load(fh)
        except (OSError, ValueError):
            result_json = {}
    code = res["exit_code"]
    if status.get("worker_error"):
        error = "helper failed: %s" % status["worker_error"]
    elif status.get("killed"):
        error = "slicer process killed: %s" % status["killed"]
    elif code != 0:
        meaning = result_json.get("error_string") or CLI_EXIT_CODES.get(code, "unknown exit code")  # type: ignore[arg-type]
        first_err = res["stderr"].splitlines()[0] if res["stderr"] else ""
        error = "Bambu Studio CLI exit code %s: %s%s" % (code, meaning, (" | " + first_err) if first_err else "")
    elif result_json.get("return_code", 0) != 0:
        error = "result.json return_code %s: %s" % (result_json.get("return_code"), result_json.get("error_string"))

    produced = os.path.join(out_dir, tmp_name)
    info: Dict[str, Any] = {}
    if error is None and not os.path.isfile(produced):
        error = "exit code 0 but no 3mf was written"
    if error is None:
        try:
            info = _read_3mf(produced, scan_gcode)
        except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as exc:
            error = "cannot read the produced 3mf: %s" % exc
    if error is None and not info.get("gcode_files"):
        error = "exit code 0 but the 3mf contains no g-code (nothing was sliced)"
    plates = result_json.get("sliced_plates") or []
    if error is None and os.path.isfile(rj_path) and not plates:
        error = "exit code 0 but result.json lists no sliced plate"

    if info:
        gcode, plate, eff = info.get("gcode", {}), info.get("plate", {}), info.get("settings") or {}
        rplate = plates[0] if plates else {}
        res["bambu_version"] = res["bambu_version"] or info.get("client_version") or gcode.get("slicer_version")
        pred = _to_float(plate.get("prediction"))
        res["print_time_s"] = int(pred) if pred is not None else (
            int(round(rplate["total_predication"])) if "total_predication" in rplate else gcode.get("total_time_s"))
        res["model_print_time_s"] = (int(round(rplate["main_predication"])) if "main_predication" in rplate
                                     else gcode.get("model_print_time_s"))
        res["print_time_text"] = gcode.get("time_text")
        weight = _to_float(plate.get("weight"))
        if weight is None and rplate.get("filaments"):
            weight = round(sum(float(f.get("total_used_g", 0.0)) for f in rplate["filaments"]), 2)
        res["filament_g"] = weight if weight is not None else gcode.get("filament_g")
        metres = [_to_float(f.get("used_m")) for f in info.get("filaments", [])]
        if metres and all(m is not None for m in metres):
            res["filament_m"] = round(sum(metres), 2)  # type: ignore[arg-type]
        elif gcode.get("filament_mm") is not None:
            res["filament_m"] = round(gcode["filament_mm"] / 1000.0, 2)
        res["layer_count"] = gcode.get("layer_count") or plate.get("_layer_count")
        res["layer_height"] = _to_float(eff.get("layer_height"))
        res["max_z_mm"] = gcode.get("max_z_mm")
        res["gcode_features"] = gcode.get("features", {})
        # --- supports: slice_info "support_used" only mirrors enable_support, so look at output
        feature_times = rplate.get("feature_type_times") or {}
        generated = (any(k.lower().startswith("support") for k in res["gcode_features"])
                     or any(k.lower().startswith("support") and v > 0 for k, v in feature_times.items())
                     or any(str(f.get("used_for_support", "")).lower() == "true" for f in info.get("filaments", [])))
        res["support_used"] = bool(generated)
        flag = eff.get("enable_support", plate.get("support_used"))
        res["support_enabled"] = str(flag).lower() in ("1", "true") if flag is not None else None
        # --- warnings: <warning msg= level= error_code=/> in slice_info.config (result.json's
        #     warning_message was always empty in testing, but is included if present)
        res["warning_details"] = [{k.strip(): v for k, v in w.items()} for w in info.get("slice_warnings", [])]
        names = [str(rplate["warning_message"])] if rplate.get("warning_message") else []
        names += [w.get("msg") or json.dumps(w) for w in res["warning_details"]]
        res["warnings"] = list(dict.fromkeys(names))
        # --- what was really used
        res["presets"] = {"process": eff.get("print_settings_id"), "machine": eff.get("printer_settings_id"),
                          "filament": eff.get("filament_settings_id"), "bed_type": eff.get("curr_bed_type"),
                          "inherits": eff.get("inherits_group")}
        for kind in _KINDS:
            for key, want in requested[kind].items():
                label = key if kind == "process" else "%s:%s" % (kind, key)
                if key not in eff:
                    res["overrides_mismatched"][label] = {"requested": want, "effective": None,
                                                         "note": "setting unknown to this Bambu Studio version"}
                elif _values_equal(want, eff[key]):
                    res["overrides_applied"][label] = eff[key]
                else:
                    res["overrides_mismatched"][label] = {"requested": want, "effective": eff[key]}
        # settings the CLI changed on its own: effective value matches none of the presets passed in
        for kind in _KINDS:
            for key, passed in configs[kind].items():
                if key in _META_KEYS or key.endswith("_settings_id") or key not in eff:
                    continue
                if any(key in requested[k] for k in _KINDS):
                    continue
                candidates = [configs[k][key] for k in _KINDS if key in configs[k]]
                if not any(_same_or_truncated(c, eff[key]) for c in candidates):
                    res["cli_adjusted_settings"].setdefault(key, {"preset": passed, "effective": eff[key]})
        if error is None:
            expect = {"print_settings_id": final_names["process"], "printer_settings_id": final_names["machine"]}
            for key, want in expect.items():
                if eff.get(key) != want:
                    error = "output was sliced with %s=%r, expected %r" % (key, eff.get(key), want)
        if error is None and strict_overrides and res["overrides_mismatched"]:
            error = "override(s) not applied: %s" % ", ".join(
                "%s (wanted %s, got %s)" % (k, v["requested"], v["effective"])
                for k, v in res["overrides_mismatched"].items())

    # ---- deliver / clean up --------------------------------------------------------------
    if error is None:
        try:
            if os.path.exists(out_3mf_path):
                os.remove(out_3mf_path)
            shutil.move(produced, out_3mf_path)
            res["out_3mf"] = out_3mf_path
        except OSError as exc:
            error = "could not move the result to %s: %s" % (out_3mf_path, exc)
    if keep_work_dir is True:
        pass                                                  # keep everything
    elif keep_work_dir is None and error is not None:
        _clean_work_dir(work_dir, remove_dir=False, diagnostics_too=False)   # keep presets + logs
    else:
        _clean_work_dir(work_dir, remove_dir=created_work_dir)
        res["work_dir"] = None
    return finish(error)


# --------------------------------------------------------------------------------------------
# command line / self test
# --------------------------------------------------------------------------------------------
def _write_test_sphere(path: str, radius: float = 15.0, n_lat: int = 48, n_lon: int = 96) -> None:
    """Binary STL of a UV sphere resting on z = 0 (pure Python, for the self test)."""
    import math
    import struct

    def pt(i: int, j: int) -> Tuple[float, float, float]:
        th = math.pi * i / n_lat
        ph = 2.0 * math.pi * (j % n_lon) / n_lon
        return (radius * math.sin(th) * math.cos(ph), radius * math.sin(th) * math.sin(ph),
                radius + radius * math.cos(th))

    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b, c, d = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            if i > 0:
                tris.append((a, b, d))
            if i < n_lat - 1:
                tris.append((b, c, d))
    with open(path, "wb") as fh:
        fh.write(b"slice_cli self-test sphere".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            u = [b[k] - a[k] for k in range(3)]
            v = [c[k] - a[k] for k in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            ln = math.sqrt(sum(x * x for x in n)) or 1.0
            fh.write(struct.pack("<12fH", n[0] / ln, n[1] / ln, n[2] / ln, *a, *b, *c, 0))


def _parse_value(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    return text


def _main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Slice an STL headlessly with the Bambu Studio CLI.")
    ap.add_argument("stl", nargs="?", help="input STL")
    ap.add_argument("out", nargs="?", help="output .gcode.3mf")
    ap.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. -s layer_height=0.16 -s enable_support=false")
    ap.add_argument("--process", default=DEFAULT_PROCESS)
    ap.add_argument("--machine", default=DEFAULT_MACHINE)
    ap.add_argument("--filament", default=DEFAULT_FILAMENT)
    ap.add_argument("--bed-type", default=DEFAULT_BED_TYPE)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the work directory (presets, logs)")
    ap.add_argument("--debug", type=int, default=3, help="CLI log level 0..5")
    ap.add_argument("--selftest", metavar="DIR", help="write a 30 mm sphere into DIR, slice it, print the result")
    args = ap.parse_args(argv)
    if args.selftest:
        os.makedirs(args.selftest, exist_ok=True)
        args.stl = os.path.join(args.selftest, "selftest_sphere30.stl")
        args.out = os.path.join(args.selftest, "selftest_sphere30.gcode.3mf")
        args.work_dir = args.work_dir or os.path.join(args.selftest, "_work")   # keep the self test in DIR
        _write_test_sphere(args.stl)
    if not args.stl or not args.out:
        ap.error("stl and out are required (or use --selftest DIR)")
    overrides = {}
    for item in args.set:
        if "=" not in item:
            ap.error("--set expects KEY=VALUE, got %r" % item)
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse_value(value.strip())
    result = slice_with_bambu(args.stl, args.out, overrides, process=args.process, machine=args.machine,
                              filament=args.filament, bed_type=args.bed_type or None, timeout_s=args.timeout,
                              work_dir=args.work_dir, keep_work_dir=True if args.keep else None,
                              debug_level=args.debug)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker":
        sys.exit(_worker_main(sys.argv[2]))
    sys.exit(_main(sys.argv[1:]))
