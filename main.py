"""Lithophane Sphere Studio - start the app, or export without the window.

    python main.py                                  open the app
    python main.py my_ornament.lsphere              open the app with a saved project
    python main.py --export out.stl [project]       build and write the model, no window
                                                    (add --no-slice to skip the Bambu Studio project)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")


def load_params(project: str | None):
    """Params from a project file, or the built-in demo ornament."""
    from lithosphere.params import MAX_IMAGES, Params, params_from_file
    from lithosphere.ui.main_window import samples_in

    if project:
        with open(project, "r", encoding="utf-8") as fh:
            return params_from_file(json.load(fh))
    params = Params()
    for k, path in enumerate(samples_in(SAMPLES)[:MAX_IMAGES]):
        params.images[k].path = path
    return params


def export_headless(out_path: str, project: str | None, slice_it: bool = True) -> int:
    from lithosphere import checks, export, geometry, slicer

    params = load_params(project)
    build = geometry.build_ornament(params)
    report = checks.verify_mesh(build.vertices, build.faces)
    if not (report["watertight"] and report["finite"] and report["volume_mm3"] > 0):
        print(f"ERROR: mesh failed its integrity check: {report}", file=sys.stderr)
        return 2
    export.write_mesh(out_path, build.vertices, build.faces, build.seam_faces)
    sheet = os.path.splitext(out_path)[0] + "_print_settings.txt"
    with open(sheet, "w", encoding="utf-8") as fh:
        fh.write(export.settings_sheet(params, build.layout, build.stats))
    sx, sy, sz = build.stats["size_mm"]
    print(f"Wrote {out_path}")
    print(f"  {sx:.1f} x {sy:.1f} x {sz:.1f} mm, {build.stats['triangles']:,} triangles, "
          f"about {build.stats['weight_g']:.0f} g of PLA")
    for problem in build.problems:
        print(f"  WARNING: {problem}")
    tips = len(checks.hanging_tips(build.vertices, build.faces))
    if tips:
        print(f"  WARNING: {tips} picture detail(s) would start in mid-air and print as blobs. "
              f"Switch Print-safe edges on.")
    else:
        print("  nothing starts in mid-air (no floating regions)")
    print(f"Wrote {sheet}")
    if slice_it and slicer.can_slice():
        target = slicer.project_path_for(out_path)
        print("Slicing with Bambu Studio ...")
        result = slicer.slice_mesh(build.vertices, build.faces, target, params,
                                   seam_faces=build.seam_faces)
        if not result.get("ok"):
            print(f"ERROR: slicing failed: {result.get('error')}", file=sys.stderr)
            return 3
        expected, unexpected = slicer.split_warnings(result.get("warnings") or [])
        print(f"Wrote {target}")
        print(f"  print time about {slicer.format_duration(result.get('print_time_s'))}, "
              f"{result.get('filament_g'):.0f} g of filament, {result.get('layer_count')} layers, "
              f"supports: {'YES' if result.get('support_used') else 'none'}")
        seam_hidden, seam_text = slicer.seam_note(result.get("seam"))
        if seam_text:
            print(f"  {'' if seam_hidden else 'WARNING: '}{seam_text}")
        rings_sound, rings_text = slicer.ring_note(result.get("seam"))
        if rings_text:
            print(f"  WARNING: {rings_text}")
        elif result.get("seam") and result["seam"].get("layers"):
            print(f"  every one of the {result['seam']['layers']} layers that carry a picture "
                  f"or a decoration is a closed ring of two wall loops")
        for text in expected:
            print(f"  expected notice: {text}")
        for text in unexpected:
            print(f"  WARNING from Bambu Studio: {text}")
    return 0


def run_gui(project: str | None) -> int:
    from lithosphere import APP_NAME
    from lithosphere.ui import style
    from PyQt6 import QtCore, QtWidgets

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("LithophaneSphereStudio")
    style.apply(app)

    data_dir = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.AppDataLocation)
    log_path = os.path.join(data_dir, "error.log")

    def on_error(kind, value, tb):
        """Never die silently: log the traceback and tell the user."""
        text = "".join(traceback.format_exception(kind, value, tb))
        try:
            os.makedirs(data_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            pass
        sys.__stderr__ and sys.__stderr__.write(text)
        QtWidgets.QMessageBox.critical(None, APP_NAME,
                                       f"Something went wrong:\n\n{value}\n\nDetails were "
                                       f"saved to {log_path}")

    sys.excepthook = on_error

    from lithosphere.ui.main_window import MainWindow

    window = MainWindow(samples_dir=SAMPLES, session_path=os.path.join(data_dir, "session.json"))
    window.show()
    # The window reopens with whatever was being worked on last time.  Opening a
    # project on top of unsaved work must ask first: that work exists nowhere else.
    if project and window._confirm_discard():
        window._load_file(project)
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Lithophane Christmas sphere generator")
    parser.add_argument("project", nargs="?", help="a saved .lsphere project")
    parser.add_argument("--export", metavar="FILE", help="write an .stl or .3mf and exit")
    parser.add_argument("--no-slice", action="store_true",
                        help="with --export: do not make the Bambu Studio project")
    args = parser.parse_args()
    out_path = os.path.abspath(args.export) if args.export else None
    project = os.path.abspath(args.project) if args.project else None
    os.chdir(HERE)
    if out_path:
        return export_headless(out_path, project, slice_it=not args.no_slice)
    args.project = project
    return run_gui(args.project)


if __name__ == "__main__":
    sys.exit(main())
