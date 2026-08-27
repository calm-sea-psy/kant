#!/usr/bin/env python3
"""Lint TIL/*.md against this repo's house formatting rules.

Usage:
    python .claude/skills/til-format/scripts/til_lint.py [FILE ...]

With no args, lints every TIL/*.md under the repo root (found by walking up
from the current directory until a TIL/ directory is seen).

Prints one line per violation as `path:line  message` and exits 1 if any.
"""
import sys, re, glob, os

try:  # Windows consoles default to cp949; force UTF-8 so Korean/em-dash print
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

MD_TABLE_SEP = re.compile(r'^\s*\|(\s*:?-+:?\s*\|)+\s*$')
# A sub-item marker glued to its text. `ㄱ)` style is unambiguous; `a.` style
# only counts when immediately followed by Hangul (else it's a sentence like
# "y.grad_fn 을 ...").
MARKER = re.compile(r'^([ㄱ-ㅎ]\)(?=\S)|[a-h]\.(?=[가-힣]))')


def find_til_dir():
    d = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(d, 'TIL')):
            return os.path.join(d, 'TIL')
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def lint(path):
    out = []
    text = open(path, encoding='utf-8').read()
    lines = text.split('\n')

    if lines and lines[0].strip() == '':
        out.append((1, '파일 첫 줄이 빈 줄 (첫 줄은 `1. ...` 이어야 함)'))

    in_fence = False
    for i, l in enumerate(lines, 1):
        stripped = l.strip()

        # fenced code blocks
        if stripped.startswith('```'):
            out.append((i, '펜스 코드 블록 ``` (코드는 4칸 들여쓰기로)'))
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # markdown headers
        if re.match(r'^#{1,6}\s', l):
            out.append((i, f'마크다운 헤더 (번호 구조로: {l.strip()[:40]})'))

        # LaTeX
        if re.search(r'(?<!\\)\$[^$\n]+\$', l):
            out.append((i, 'LaTeX $...$ (평문 기호나 말로 풀어쓰기)'))

        # marker without following space
        if MARKER.match(l):
            out.append((i, f'마커 뒤 공백 누락: {l[:20]!r}'))

    # markdown table checks
    for i, l in enumerate(lines):
        if l.lstrip().startswith('| ') and '|' in l[l.index('|') + 1:]:
            # is this the header row of a table? next non-... line must be a sep
            is_header = (i + 1 < len(lines) and MD_TABLE_SEP.match(lines[i + 1]))
            is_body = (i > 0 and (MD_TABLE_SEP.match(lines[i - 1]) or lines[i - 1].lstrip().startswith('|')))
            if not is_header and not is_body:
                out.append((i + 1, '마크다운 표 행인데 |---| 구분선이 없음'))
            if is_header:
                if i > 0 and lines[i - 1].strip() != '':
                    out.append((i + 1, '표 앞에 빈 줄 없음'))
                j = i + 2
                while j < len(lines) and lines[j].lstrip().startswith('|'):
                    j += 1
                if j < len(lines) and lines[j].strip() != '':
                    out.append((j + 1, '표 뒤에 빈 줄 없음'))

    return out


def main(argv):
    files = argv[1:]
    if not files:
        td = find_til_dir()
        if not td:
            print('TIL/ 디렉토리를 찾을 수 없습니다', file=sys.stderr)
            return 2
        files = sorted(glob.glob(os.path.join(td, '*.md')))

    total = 0
    for f in files:
        for ln, msg in lint(f):
            print(f'{f}:{ln}  {msg}')
            total += 1
    if total:
        print(f'\n{total}건 위반')
        return 1
    print('OK — 위반 없음')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
