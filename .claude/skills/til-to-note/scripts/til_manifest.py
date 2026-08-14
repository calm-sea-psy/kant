#!/usr/bin/env python3
"""
TIL -> NOTE manifest helper.

The manifest (NOTE/.manifest.json) tracks, for every TIL/*.md file, the sha256
hash of its content at the time it was last incorporated into a NOTE group
file, plus which group file it landed in. This script does the mechanical,
exactly-verifiable part (hashing, diffing) so the model only has to do the
part that actually requires judgment: classifying content into topic groups
and rewriting prose.

Usage:
    python til_manifest.py scan [--til-dir TIL] [--manifest NOTE/.manifest.json]
        Prints a JSON report of {new: [...], changed: [...], unchanged: [...]}
        comparing current TIL/*.md files against the manifest.

    python til_manifest.py update <til_filename> <group_slug> [--til-dir TIL] [--manifest NOTE/.manifest.json]
        Records that <til_filename> (e.g. 260814.md) has been incorporated
        into group <group_slug> (e.g. 06-평가지표와데이터누수), stamping the
        current content hash and timestamp. Call this once per TIL file,
        right after you've finished writing its content into the
        summary/detail group files.

    python til_manifest.py init --map MAPPING_JSON [--til-dir TIL] [--manifest NOTE/.manifest.json]
        Bootstraps a manifest from scratch given a mapping of
        {til_filename: group_slug}. Use this once, the first time this skill
        is set up on a repo that already has hand-written NOTE content (as
        this repo does).

Exit codes: 0 on success, 1 on usage/argument errors.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"til_files": {}}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cmd_scan(args):
    til_dir = Path(args.til_dir)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    known = manifest.get("til_files", {})

    current_files = sorted(p.name for p in til_dir.glob("*.md"))

    new_files = []
    changed_files = []
    unchanged_files = []

    for name in current_files:
        h = sha256_of(til_dir / name)
        if name not in known:
            new_files.append(name)
        elif known[name]["hash"] != h:
            changed_files.append({"file": name, "previous_group": known[name].get("group")})
        else:
            unchanged_files.append(name)

    missing_files = sorted(set(known.keys()) - set(current_files))

    report = {
        "new": new_files,
        "changed": changed_files,
        "unchanged": unchanged_files,
        "missing": missing_files,  # in manifest but no longer on disk - worth a human's attention
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_update(args):
    til_dir = Path(args.til_dir)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    til_path = til_dir / args.til_filename
    if not til_path.exists():
        print(f"error: {til_path} does not exist", file=sys.stderr)
        return 1

    manifest.setdefault("til_files", {})[args.til_filename] = {
        "hash": sha256_of(til_path),
        "group": args.group_slug,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_manifest(manifest_path, manifest)
    print(f"recorded {args.til_filename} -> {args.group_slug}")
    return 0


def cmd_init(args):
    til_dir = Path(args.til_dir)
    manifest_path = Path(args.manifest)

    if manifest_path.exists() and not args.force:
        print(f"error: {manifest_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    manifest = {"til_files": {}}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for til_filename, group_slug in mapping.items():
        til_path = til_dir / til_filename
        if not til_path.exists():
            print(f"warning: {til_path} not found, skipping", file=sys.stderr)
            continue
        manifest["til_files"][til_filename] = {
            "hash": sha256_of(til_path),
            "group": group_slug,
            "processed_at": now,
        }

    save_manifest(manifest_path, manifest)
    print(f"initialized {manifest_path} with {len(manifest['til_files'])} entries")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--til-dir", default="TIL")
    parser.add_argument("--manifest", default="NOTE/.manifest.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan")

    p_update = sub.add_parser("update")
    p_update.add_argument("til_filename")
    p_update.add_argument("group_slug")

    p_init = sub.add_parser("init")
    p_init.add_argument("--map", required=True, help="path to a JSON file mapping {til_filename: group_slug}")
    p_init.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "scan":
        sys.exit(cmd_scan(args))
    elif args.cmd == "update":
        sys.exit(cmd_update(args))
    elif args.cmd == "init":
        sys.exit(cmd_init(args))


if __name__ == "__main__":
    main()
