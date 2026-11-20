# -*- coding: utf-8 -*-
"""把章节里成对的 ASCII 直引号换成中文引号。

用法：
    python books/build/textnorm.py books/concepts/chapters
    python books/build/textnorm.py books/concepts/chapters --check

只在正文行里替换，跳过代码块与行内代码。
"""

import argparse
import re
import sys
from pathlib import Path

FENCE = re.compile(r"^\s*(```+|~~~+)")


def convert_line(line: str) -> str:
    parts = re.split(r"(`[^`]*`)", line)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        out = []
        opening = True
        for ch in seg:
            if ch == '"':
                out.append("“" if opening else "”")
                opening = not opening
            else:
                out.append(ch)
        parts[i] = "".join(out)
    return "".join(parts)


def process(path: Path, check: bool) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    in_code = False
    changed = 0
    for i, line in enumerate(lines):
        if FENCE.match(line):
            in_code = not in_code
            continue
        if in_code or '"' not in line:
            continue
        new = convert_line(line)
        if new != line:
            changed += 1
            if not check:
                lines[i] = new
    if changed and not check:
        path.write_text("\n".join(lines), encoding="utf-8")
    if changed:
        print("%s  %d 行" % (path.name, changed))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--check", action="store_true", help="只报告，不修改")
    args = ap.parse_args()

    root = Path(args.target)
    files = sorted(root.glob("*.md")) if root.is_dir() else [root]
    total = sum(process(f, args.check) for f in files)
    print("共 %d 行需要处理" % total)


if __name__ == "__main__":
    main()
