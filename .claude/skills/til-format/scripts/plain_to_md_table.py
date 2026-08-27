#!/usr/bin/env python3
"""Convert space-aligned plain-text tables in TIL/*.md to markdown tables.

Usage:
    python .claude/skills/til-format/scripts/plain_to_md_table.py [FILE ...]

With no args, processes every TIL/*.md (walks up for a TIL/ dir).

HEURISTIC — review the diff afterward. A "table" here is a run of >=2
consecutive lines that:
  - start at column 0 (indented lines are code, skipped)
  - each split into >=2 cells on runs of 2+ spaces
  - contain no box-drawing / bar / emoji chars (those are diagrams)
  - whose first row does not look like a definition list (`name : value`)

Rows with more cells than the header are merged into the last cell; fewer
are right-padded.
"""
import sys, re, glob, os

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

DIAG = set('─│┌┐└┘├┤┬┴┼╲╱╳═╪╬▶█▐▌＿')  # NOTE: arrows ↑↓→← are fine inside real cells


def find_til_dir():
    d = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(d, 'TIL')):
            return os.path.join(d, 'TIL')
        p = os.path.dirname(d)
        if p == d:
            return None
        d = p


def is_diagram(line):
    if any(c in DIAG for c in line):
        return True
    if any(e in line for e in ('🔴', '🔵', '🟢', '🟡')):
        return True
    if re.search(r'──|←─|─→|→→|←←', line):
        return True
    return line.lstrip()[:1] in '├└│┌'


def cells(line):
    return [c.strip() for c in re.split(r'\s{2,}', line.rstrip()) if c.strip()]


def defn_list(header):
    return any(c[:1] in ':=+' or c[:2] in ('->', '=>') for c in header[1:])


def convert(path):
    lines = open(path, encoding='utf-8').read().split('\n')
    out, i, n, changed = [], 0, len(lines), 0
    while i < n:
        line = lines[i]
        prev_blank = (not out) or out[-1].strip() == ''
        if (prev_blank and line and not line.startswith((' ', '\t'))
                and not is_diagram(line) and len(cells(line)) >= 2):
            j, block = i, []
            while (j < n and lines[j].strip() and not lines[j].startswith((' ', '\t'))
                   and not is_diagram(lines[j])):
                c = cells(lines[j])
                if len(c) < 2:
                    break
                block.append(c)
                j += 1
            if len(block) >= 2 and not defn_list(block[0]):
                ncol = len(block[0])

                def norm(c):
                    if len(c) > ncol:
                        return c[:ncol - 1] + ['  '.join(c[ncol - 1:])]
                    return c + [''] * (ncol - len(c))

                out.append('| ' + ' | '.join(norm(block[0])) + ' |')
                out.append('|' + '|'.join(['---'] * ncol) + '|')
                for row in block[1:]:
                    out.append('| ' + ' | '.join(norm(row)) + ' |')
                changed += 1
                i = j
                continue
        out.append(line)
        i += 1
    return '\n'.join(out), changed


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
        new, c = convert(f)
        if c:
            open(f, 'w', encoding='utf-8', newline='\n').write(new)
            print(f'{os.path.basename(f)}: {c}개 표 변환')
            total += c
    print(f'총 {total}개. git diff 로 검토하세요.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
