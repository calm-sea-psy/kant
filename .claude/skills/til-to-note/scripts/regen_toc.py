#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the "## 목차" block of a NOTE domain file from its headings.

NOTE domain files (NOTE/1-수학.md, 2-머신러닝.md, 3-딥러닝.md, 4-LLM.md) carry a
table of contents near the top:

    ## 목차

    - [토픽 제목](#anchor)
        - [1. 절 제목](#anchor)
        ...

    ---

The anchors must match GitHub's heading-slug algorithm exactly, and the list
must stay in sync when sections are added or renamed. This script rebuilds that
block in place. Level-2 headings (## topic) and level-3 headings (### section)
are listed; deeper headings and the "## 목차" heading itself are skipped.

Usage:
    python .claude/skills/til-to-note/scripts/regen_toc.py NOTE/3-딥러닝.md
    python .claude/skills/til-to-note/scripts/regen_toc.py --check NOTE/1-수학.md NOTE/2-머신러닝.md NOTE/3-딥러닝.md NOTE/4-LLM.md

--check exits 1 (and prints which files are stale) without modifying anything.
Without --check, files are rewritten in place.
"""
import argparse
import collections
import pathlib
import re
import sys

# GitHub heading anchors: lowercase, remove every char that is not a Unicode
# word char / hyphen-minus / space, then spaces -> "-". Duplicate slugs get
# "-1", "-2", ... in document order. Matches GitHub's TableOfContentsFilter
# (Ruby: text.downcase.gsub(/[^\p{Word}\- ]/u, '').tr(' ', '-')).
_SLUG_RE = re.compile(r"[^\w\- ]", re.UNICODE)
_FENCE = re.compile(r"^\s*`{3,}")
_HEAD = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def make_slug():
    seen = collections.defaultdict(int)

    def slug(text):
        s = _SLUG_RE.sub("", text.strip().lower()).replace(" ", "-")
        n = seen[s]
        seen[s] += 1
        return s if n == 0 else f"{s}-{n}"

    return slug


def collect_headings(lines):
    """[(level, text)] for ATX headings outside fenced code."""
    out = []
    in_fence = False
    for ln in lines:
        if _FENCE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEAD.match(ln)
        if m:
            out.append((len(m.group(1)), m.group(2).strip()))
    return out


def build_toc(lines):
    slug = make_slug()
    entries = ["## 목차", ""]
    for level, text in collect_headings(lines):
        anchor = slug(text)  # advance for EVERY heading (matches GitHub ordering)
        if text == "목차":
            continue
        if level == 2:
            entries.append(f"- [{text}](#{anchor})")
        elif level == 3:
            entries.append(f"    - [{text}](#{anchor})")
    return entries


def splice_toc(text):
    """Replace the '## 목차' block (from its heading to the next '---' line)."""
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "## 목차")
    except StopIteration:
        raise SystemExit("no '## 목차' heading found")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise SystemExit("no '---' separator after '## 목차'")
    return "\n".join(lines[:start] + build_toc(lines) + [""] + lines[end:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true", help="report stale TOCs, do not write")
    args = ap.parse_args()

    stale = 0
    for fp in args.files:
        p = pathlib.Path(fp)
        old = p.read_text(encoding="utf-8")
        new = splice_toc(old)
        if new == old:
            print(f"ok    {fp}")
        elif args.check:
            stale += 1
            print(f"STALE {fp}")
        else:
            p.write_text(new, encoding="utf-8", newline="\n")
            print(f"wrote {fp}")

    if args.check and stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
