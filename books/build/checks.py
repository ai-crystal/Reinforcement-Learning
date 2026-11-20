# -*- coding: utf-8 -*-
"""书稿体检。构建前先跑一遍，把结构和格式问题挡在排版之前。

用法：
    python books/build/checks.py --book books/concepts

检查项：
    1. manifest 里列出的文件都存在，章节以 `## ` 开头
    2. 插图引用能解析，目录号与章号一致
    3. 代码块必须有语言标注，且不能嵌套
    4. 表题编号按章连续，图号引用不超过实际插图数
    5. 标题里没有会让 LaTeX 报错的字符
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BAD_HEADING = set("%&#_$\\{}~^")
FENCE = re.compile(r"^\s*(```+|~~~+)\s*(\S*)")


def check(book: Path) -> list[str]:
    errors: list[str] = []
    manifest = book / "manifest.txt"
    if not manifest.exists():
        return ["缺少 manifest.txt"]

    entries = [ln.strip() for ln in manifest.read_text(encoding="utf-8").splitlines()]
    entries = [e for e in entries if e and not e.startswith("#")]

    chapters = []
    for rel in entries:
        p = book / rel
        if not p.exists():
            errors.append("manifest 列出的文件不存在：%s" % rel)
            continue
        if rel.startswith("chapters/"):
            chapters.append((rel, p))

    for idx, (rel, path) in enumerate(chapters, start=1):
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        name = path.name

        # 章号一致性（chapter 目录下按顺序编号）
        m = re.match(r"ch(\d+)\.md$", name)
        if m and int(m.group(1)) != idx:
            errors.append("%s 的文件编号与它在 manifest 中的位置（第 %d 章）不一致" % (name, idx))

        if not lines or not lines[0].startswith("## "):
            errors.append("%s 第一行必须是 `## 章标题`" % name)

        # 代码块
        in_code = False
        lang = ""
        for no, line in enumerate(lines, start=1):
            mf = FENCE.match(line)
            if mf:
                if not in_code:
                    in_code = True
                    lang = mf.group(2)
                    if not lang:
                        errors.append("%s:%d 代码块缺少语言标注" % (name, no))
                    if line.strip().startswith("````"):
                        in_code = "nested"
                else:
                    if in_code == "nested":
                        errors.append("%s:%d 出现嵌套代码块，会破坏排版" % (name, no))
                    in_code = False
                    lang = ""
            elif in_code and line.strip().startswith("```") and len(line.strip()) > 3:
                errors.append("%s:%d 四重围栏，检查是否嵌套" % (name, no))
        if in_code:
            errors.append("%s 有未闭合的代码块" % name)

        # 标题字符
        for no, line in enumerate(lines, start=1):
            if re.match(r"^#{1,4} \S", line):
                body = re.sub(r"^#+\s*", "", line)
                bad = sorted({c for c in body if c in BAD_HEADING})
                if bad:
                    errors.append("%s:%d 标题含特殊字符 %s" % (name, no, "".join(bad)))

        # 插图
        imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        for ref in imgs:
            if not ref.startswith("../images/"):
                errors.append("%s 的插图没有写在 ../images/ 下：%s" % (name, ref))
                continue
            parts = ref.split("/")
            if len(parts) < 4 or not parts[2].isdigit():
                errors.append("%s 的插图路径格式不对：%s" % (name, ref))
                continue
            if int(parts[2]) != idx:
                errors.append("%s 引用了 %s 章的插图目录：%s" % (name, parts[2], ref))
            if not (path.parent / ref).resolve().exists():
                errors.append("%s 引用的插图不存在：%s" % (name, ref))

        # 图号引用
        for sub, num in re.findall(r"图\s*(\d+)-(\d+)", text):
            if int(sub) != idx:
                errors.append("%s 正文引用了别章的图号：图 %s-%s" % (name, sub, num))
            elif int(num) > len(imgs):
                errors.append("%s 引用了不存在的图：图 %s-%s（本章只有 %d 张插图）"
                              % (name, sub, num, len(imgs)))

        # 表题连续性
        caps = re.findall(r"::: \{\.tabcap\}\s*\n表\s*(\d+)-(\d+)", text)
        for i, (sub, num) in enumerate(caps, start=1):
            if int(sub) != idx:
                errors.append("%s 的表题编号与章号不一致：表 %s-%s" % (name, sub, num))
            if int(num) != i:
                errors.append("%s 的表题编号不连续：第 %d 个表写成了表 %s-%s" % (name, i, sub, num))

    return errors


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="")
    ap.add_argument("--all", action="store_true", help="体检 books/ 下所有册")
    args = ap.parse_args()

    if not args.all and not args.book:
        raise SystemExit("用法：checks.py --book books/<册名> 或 --all")
    targets = ([p.parent for p in sorted((ROOT / "books").glob("*/book.json"))]
               if args.all else [(ROOT / args.book).resolve()])

    total = 0
    for book in targets:
        errors = check(book)
        name = book.name
        if errors:
            total += len(errors)
            print("[%s] %d 个问题" % (name, len(errors)))
            for e in errors:
                print("   -", e)
        else:
            print("[%s] 通过" % name)
    if total:
        sys.exit(1)


if __name__ == "__main__":
    main()
