#!/usr/bin/env python3
"""Manage enrolled faces in known_faces/.

Commands:
  list                 - list enrolled faces with metadata
  info NAME            - show detailed info about an enrolled face
  remove NAME          - remove a named enrollment
  rename OLD NEW       - rename an enrolled face
  clear [--yes]        - remove all enrolled faces
  export ARCHIVE.zip   - export all face encodings & images to a zip archive
  import ARCHIVE.zip   - import face encodings & images from a zip archive
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import List, Optional
import zipfile

import numpy as np

from utils import sanitize_name

DEFAULT_KNOWN_DIR = Path(__file__).parent / "known_faces"


def get_known_dir(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        return Path(custom_path)
    return DEFAULT_KNOWN_DIR


def list_faces(known_dir: Optional[Path] = None, json_output: bool = False) -> List[str]:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    if not k_dir.exists():
        if json_output:
            print(json.dumps([]))
        else:
            print("No known faces directory found.")
        return []

    npy_files = sorted(k_dir.glob("*.npy"))
    if not npy_files:
        if json_output:
            print(json.dumps([]))
        else:
            print("No enrolled faces.")
        return []

    results = []
    for p in npy_files:
        name = p.stem
        jpg_p = p.with_suffix(".jpg")
        has_jpg = jpg_p.exists()
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        size_kb = p.stat().st_size / 1024.0
        results.append({
            "name": name,
            "has_image": has_jpg,
            "modified": mtime,
            "size_kb": round(size_kb, 2),
        })

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print(f"Enrolled faces ({len(results)} total):")
        for item in results:
            img_tag = "[img]" if item["has_image"] else "     "
            print(f"  • {item['name']:<20} {img_tag}  (enrolled: {item['modified']})")

    return [r["name"] for r in results]


def info_face(name: str, known_dir: Optional[Path] = None, json_output: bool = False) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    clean_name = sanitize_name(name)
    npy_path = k_dir / f"{clean_name}.npy"
    jpg_path = k_dir / f"{clean_name}.jpg"

    if not npy_path.exists():
        print(f"Error: Face '{clean_name}' not found.", file=sys.stderr)
        return False

    try:
        enc = np.load(npy_path)
        shape_info = list(enc.shape)
        dtype_info = str(enc.dtype)
    except Exception as e:
        shape_info = []
        dtype_info = f"Error reading array: {e}"

    stat = npy_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    info = {
        "name": clean_name,
        "encoding_file": str(npy_path),
        "encoding_shape": shape_info,
        "encoding_dtype": dtype_info,
        "has_image": jpg_path.exists(),
        "image_file": str(jpg_path) if jpg_path.exists() else None,
        "enrolled_at": mtime,
    }

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        print(f"Face Profile: {clean_name}")
        print(f"  • Enrolled at:    {mtime}")
        print(f"  • Vector shape:   {shape_info} ({dtype_info})")
        print(f"  • Encoding path:  {npy_path}")
        print(f"  • Image path:     {jpg_path if jpg_path.exists() else 'None'}")

    return True


def remove_face(name: str, known_dir: Optional[Path] = None) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    clean_name = sanitize_name(name)
    npy_path = k_dir / f"{clean_name}.npy"
    jpg_path = k_dir / f"{clean_name}.jpg"

    removed = False
    for p in (npy_path, jpg_path):
        if p.exists():
            p.unlink()
            removed = True

    if removed:
        print(f"Successfully removed enrollment for '{clean_name}'")
        return True
    else:
        print(f"No enrollment found for '{clean_name}'", file=sys.stderr)
        return False


def rename_face(old_name: str, new_name: str, known_dir: Optional[Path] = None) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    clean_old = sanitize_name(old_name)
    clean_new = sanitize_name(new_name)

    old_npy = k_dir / f"{clean_old}.npy"
    old_jpg = k_dir / f"{clean_old}.jpg"
    new_npy = k_dir / f"{clean_new}.npy"
    new_jpg = k_dir / f"{clean_new}.jpg"

    if not old_npy.exists():
        print(f"Error: Original face '{clean_old}' not found.", file=sys.stderr)
        return False

    if new_npy.exists():
        print(f"Error: Target name '{clean_new}' already exists.", file=sys.stderr)
        return False

    old_npy.rename(new_npy)
    if old_jpg.exists():
        old_jpg.rename(new_jpg)

    print(f"Successfully renamed '{clean_old}' to '{clean_new}'")
    return True


def clear_faces(known_dir: Optional[Path] = None, force: bool = False) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    if not k_dir.exists():
        print("Nothing to clear.")
        return True

    files = list(k_dir.glob("*.npy")) + list(k_dir.glob("*.jpg"))
    if not files:
        print("No faces to clear.")
        return True

    if not force:
        resp = input(f"Are you sure you want to remove all {len(files)} face files? [y/N]: ")
        if resp.lower() not in ("y", "yes"):
            print("Aborted.")
            return False

    for f in files:
        f.unlink()

    print(f"Cleared {len(files)} files from {k_dir}.")
    return True


def export_faces(archive_path: str, known_dir: Optional[Path] = None) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    if not k_dir.exists():
        print("Error: No known faces directory to export.", file=sys.stderr)
        return False

    files = list(k_dir.glob("*.npy")) + list(k_dir.glob("*.jpg"))
    if not files:
        print("Error: No faces to export.", file=sys.stderr)
        return False

    target_zip = Path(archive_path)
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    print(f"Exported {len(files)} files to '{target_zip}'")
    return True


def import_faces(archive_path: str, known_dir: Optional[Path] = None) -> bool:
    k_dir = known_dir or DEFAULT_KNOWN_DIR
    src_zip = Path(archive_path)
    if not src_zip.exists():
        print(f"Error: Archive '{src_zip}' not found.", file=sys.stderr)
        return False

    k_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(src_zip, "r") as zf:
        for member in zf.infolist():
            # Security: avoid zip slip
            fname = Path(member.filename).name
            if fname.endswith(".npy") or fname.endswith(".jpg"):
                zf.extract(member, k_dir)
                count += 1

    print(f"Imported {count} files into '{k_dir}'")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage enrolled faces in known_faces/")
    parser.add_argument("--dir", type=str, default=None, help="Custom directory path for known faces")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    subparsers.add_parser("list", help="List enrolled names and status")

    # info
    p_info = subparsers.add_parser("info", help="Show details of a specific enrolled face")
    p_info.add_argument("name", help="Name of the person")

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove an enrolled face")
    p_remove.add_argument("name", help="Name of the person to remove")

    # rename
    p_rename = subparsers.add_parser("rename", help="Rename an enrolled face")
    p_rename.add_argument("old_name", help="Current name")
    p_rename.add_argument("new_name", help="New name")

    # clear
    p_clear = subparsers.add_parser("clear", help="Clear all enrolled faces")
    p_clear.add_argument("--yes", "-y", action="store_true", help="Confirm deletion without prompt")

    # export
    p_export = subparsers.add_parser("export", help="Export enrolled faces to a zip archive")
    p_export.add_argument("archive", help="Destination zip file path")

    # import
    p_import = subparsers.add_parser("import", help="Import enrolled faces from a zip archive")
    p_import.add_argument("archive", help="Source zip file path")

    args = parser.parse_args()
    k_dir = get_known_dir(args.dir)

    if args.command == "list":
        list_faces(k_dir, json_output=args.json)
    elif args.command == "info":
        success = info_face(args.name, k_dir, json_output=args.json)
        if not success:
            sys.exit(1)
    elif args.command == "remove":
        success = remove_face(args.name, k_dir)
        if not success:
            sys.exit(1)
    elif args.command == "rename":
        success = rename_face(args.old_name, args.new_name, k_dir)
        if not success:
            sys.exit(1)
    elif args.command == "clear":
        clear_faces(k_dir, force=args.yes)
    elif args.command == "export":
        success = export_faces(args.archive, k_dir)
        if not success:
            sys.exit(1)
    elif args.command == "import":
        success = import_faces(args.archive, k_dir)
        if not success:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
