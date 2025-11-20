#!/usr/bin/env python3
"""Manage enrolled faces in known_faces/.

Commands:
  list        - list enrolled names
  remove NAME - remove a named enrollment
"""
import argparse
from pathlib import Path
import os


KNOWN_DIR = Path(__file__).parent / "known_faces"


def list_faces():
    if not KNOWN_DIR.exists():
        print("No known faces directory yet.")
        return
    files = sorted([p.stem for p in KNOWN_DIR.iterdir() if p.suffix.lower() == ".npy"])
    if not files:
        print("No enrolled faces.")
        return
    print("Enrolled faces:")
    for f in files:
        print(" -", f)


def remove_face(name: str):
    name_s = "_".join(name.strip().split())
    npy = KNOWN_DIR / f"{name_s}.npy"
    jpg = KNOWN_DIR / f"{name_s}.jpg"
    removed = False
    for p in (npy, jpg):
        if p.exists():
            p.unlink()
            removed = True
    if removed:
        print(f"Removed enrollment for '{name_s}'")
    else:
        print(f"No enrollment found for '{name_s}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "remove"], help="Command")
    parser.add_argument("name", nargs="?", help="Name for remove command")
    args = parser.parse_args()
    if args.command == "list":
        list_faces()
    elif args.command == "remove":
        if not args.name:
            print("Please specify a name to remove")
            return
        remove_face(args.name)


if __name__ == "__main__":
    main()
