# -*- coding: utf-8 -*-
"""根据各册的 manifest 生成 docsify 侧栏 _sidebar.md。

章节标题取自每章第一行的 `## 标题`，部标题取自 front/partN.md 的第一行。
站内链接一律写成仓库绝对路径，避免相对路径在 docsify 里解析错。

用法： python books/build/make_sidebars.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOKS = ROOT / "books"

VOLUMES = [
    ("concepts", "概念篇"),
    ("hands-on", "动手篇"),
    ("algorithms", "算法篇"),
    ("llm-rl", "大模型篇"),
]


def first_line(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line)
    return path.stem


def build(vol: str, label: str) -> str:
    base = BOOKS / vol
    entries = [ln.strip() for ln in (base / "manifest.txt").read_text(encoding="utf-8").splitlines()]
    entries = [e for e in entries if e and not e.startswith("#")]

    lines = ["- [本册首页](books/%s/README.md)" % vol, ""]
    part_no = 0
    for rel in entries:
        path = base / rel
        title = first_line(path)
        if rel.startswith("front/part"):
            part_no += 1
            lines.append("- **%s**" % title)
        elif rel.startswith("chapters/"):
            num = int(re.search(r"ch(\d+)", rel).group(1))
            lines.append("  - [第 %d 章　%s](books/%s/%s)" % (num, title, vol, rel))
    lines.append("")
    lines.append("- [下载本册 PDF](output/%s ':ignore')"
                 % json.loads((base / "book.json").read_text(encoding="utf-8"))["pdf"])
    return "\n".join(lines) + "\n"


def main() -> None:
    for vol, label in VOLUMES:
        text = build(vol, label)
        (BOOKS / vol / "_sidebar.md").write_text(text, encoding="utf-8")
        print("sidebar:", vol, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
