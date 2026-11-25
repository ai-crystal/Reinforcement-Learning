#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 books/ 下某一册编译成 PDF。

用法：
    python books/build/build.py --book books/concepts
    python books/build/build.py --book books/hands-on --no-opt

一册的目录约定：
    book.json     元数据（书名、副标题、输出文件名……）
    manifest.txt  正文顺序（含部扉页与附录）
    front.md      前言
    afterword.md  后记（可选）
    chapters/     章节 Markdown
    images/       插图
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOKS = ROOT / "books"
BUILD = BOOKS / "build"
TEMPLATE = BUILD / "template" / "book.tex"

EXTRA_PATH = [
    Path(r"E:\tools\TinyTeX\bin\windows"),
    Path(r"E:\tools\pandoc-3.12.1"),
    Path(r"C:\Windows\System32"),
]


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in EXTRA_PATH:
        for suffix in (".exe", ".bat", ""):
            p = d / (name + suffix)
            if p.exists():
                return str(p)
    return None


def run(cmd, cwd: Path, quiet: bool = False) -> None:
    print("  >", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(p) for p in EXTRA_PATH] + [env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=str(cwd), env=env,
                          stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        if quiet and proc.stdout:
            print("\n".join(proc.stdout.splitlines()[-40:]))
        raise SystemExit("Command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))


def part_to_latex(md: str) -> str:
    """部扉页：Markdown 里写 `# 第一部　试错`，转换成 LaTeX 的部标题页。"""
    lines = md.split("\n")
    m = re.match(r"#\s*第(.+?)部[　 ]+(.*)", lines[0].strip())
    if not m:
        return md
    num, title = m.group(1), m.group(2).strip()
    body = "\n".join(lines[1:]).strip()
    return ("```{=latex}\n\\partpage{%s}{%s}\n```\n\n"
            "::: {.partintro}\n%s\n:::\n" % (num, title, body))


def assemble(book: Path, manifest: Path, images: Path) -> str:
    parts = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = book / line
        if not p.exists():
            raise SystemExit("manifest 里列出的文件不存在：%s" % p)
        text = p.read_text(encoding="utf-8")
        if p.parent.name == "front" and p.name.startswith("part"):
            text = part_to_latex(text)
        parts.append(text)
    body = "\n\n".join(parts)
    # Markdown 里用相对本章的 ../images/NN/x.png，编译时换成绝对路径
    body = body.replace("](../images/", "](%s/" % images.as_posix())
    return body


def fragment(src: Path, out: Path, title: str, pandoc: str) -> None:
    """前言、后记不带章号，单独转成 LaTeX 片段。"""
    run([pandoc, str(src), "-f", "markdown-smart+tex_math_dollars+pipe_tables+fenced_divs",
         "-t", "latex", "-o", str(out)], ROOT, quiet=True)
    head = ("\\chapter*{%s}\n\\addcontentsline{toc}{chapter}{%s}\n"
            "\\markboth{%s}{}\n\\thispagestyle{plain}\n" % (title, title, title))
    out.write_text(head + out.read_text(encoding="utf-8"), encoding="utf-8")


def build_once(pandoc: str, xelatex: str, work: Path, body: str, images: Path,
               meta: dict, tag: str) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    (work / "body.md").write_text(body, encoding="utf-8")
    tex = work / ("book%s.tex" % tag)
    tpl_text = TEMPLATE.read_text(encoding="utf-8")
    repl = {
        "@@IMAGES@@": images.as_posix(),
        "@@COVERART@@": (images / "cover" / "cover-art.jpg").as_posix(),
        "@@BACKART@@": (images / "cover" / "back-art.jpg").as_posix(),
        "@@TITLE@@": meta.get("title", "试错"),
        "@@SERIES@@": meta.get("series", "试错"),
        "@@VOLUME@@": meta.get("volume", "试错"),
        "@@SUBTITLE@@": meta.get("subtitle", ""),
        "@@VOLNO@@": meta.get("volno", "01"),
        "@@VOLTOTAL@@": str(meta.get("voltotal", 4)),
        "@@YEAR@@": str(meta.get("year", 2026)),
        "@@SUBJECT@@": meta.get("subject", "强化学习"),
        "@@KEYWORDS@@": meta.get("keywords", "强化学习, 强化学习, PPO, RLHF, GRPO"),
        "@@BLURB@@": meta.get("blurb", ""),
        "@@TOCDEPTH@@": str(meta.get("toc_depth", 1)),
        "@@ENNAME@@": meta.get("en", "REINFORCEMENT LEARNING"),
        "@@ACCENT@@": meta.get("accent", "1E5C4F"),
    }
    for k, v in repl.items():
        tpl_text = tpl_text.replace(k, str(v))
    tpl = work / ("template%s.tex" % tag)
    tpl.write_text(tpl_text, encoding="utf-8")
    run([pandoc, str(work / "body.md"),
         "-f", "markdown-smart+tex_math_dollars+pipe_tables+fenced_divs+raw_tex",
         "--top-level-division=part", "--number-sections",
         "--lua-filter", str(BUILD / "boxes.lua"),
         "--template", str(tpl), "-o", str(tex)], ROOT, quiet=True)
    for _ in (1, 2):
        run([xelatex, "-interaction=nonstopmode", "-halt-on-error", tex.name], work,
            quiet=True)
    pdf = work / ("book%s.pdf" % tag)
    if not pdf.exists():
        raise SystemExit("xelatex 没有生成 PDF，检查 %s/book%s.log" % (work, tag))
    return pdf


def shrink_images(body: str, images: Path, work: Path) -> str:
    from PIL import Image
    jpg_dir = work / "img_jpg"
    jpg_dir.mkdir(exist_ok=True)
    out = body
    for png in sorted(images.rglob("*.png")):
        rel = png.relative_to(images)
        dst = jpg_dir / rel.with_suffix(".jpg")
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(png).convert("RGB")
        if im.width > 1200:
            im = im.resize((1200, int(im.height * 1200 / im.width)), Image.LANCZOS)
        im.save(dst, quality=82, optimize=True)
        out = out.replace(png.as_posix(), dst.as_posix())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="册目录，例如 books/concepts")
    ap.add_argument("--no-opt", action="store_true", help="不压缩插图")
    args = ap.parse_args()

    book = (ROOT / args.book).resolve()
    if not book.is_dir():
        raise SystemExit("找不到这一册：%s" % book)

    meta = json.loads((book / "book.json").read_text(encoding="utf-8"))
    images = book / "images"
    work = BUILD / "work" / book.name
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    pandoc = find_tool("pandoc")
    xelatex = find_tool("xelatex")
    if not pandoc or not xelatex:
        raise SystemExit("缺少 pandoc 或 xelatex，检查 E:\\tools 下的安装。")

    print("[1/4] 拼装 Markdown")
    body = assemble(book, book / "manifest.txt", images)
    front = book / "front.md"
    afterword = book / "afterword.md"
    fragment(front, work / "front.tex", "前　言", pandoc)
    if afterword.exists():
        fragment(afterword, work / "afterword.tex", "后　记", pandoc)
    else:
        (work / "afterword.tex").write_text("", encoding="utf-8")

    print("[2/4] pandoc -> LaTeX")
    pdf = build_once(pandoc, xelatex, work, body, images, meta, "")

    size_mb = pdf.stat().st_size / 1024 / 1024
    print("[3/4] 第一次编译 %.1f MB" % size_mb)
    if size_mb > 19.0 and not args.no_opt:
        print("      超过 19 MB，压缩插图后重排")
        body2 = shrink_images(body, images, work)
        pdf = build_once(pandoc, xelatex, work, body2, images, meta, "-c")
        size_mb = pdf.stat().st_size / 1024 / 1024

    output = ROOT / "output" / meta.get("pdf", book.name + ".pdf")
    output.parent.mkdir(exist_ok=True)
    try:
        shutil.copyfile(pdf, output)
    except PermissionError:
        alt = output.with_name(output.stem + "-new" + output.suffix)
        shutil.copyfile(pdf, alt)
        print("提示：%s 正被其他程序占用，本次输出到 %s" % (output.name, alt.name))
        output = alt
    pages = pdf_pages(pdf)
    print("[4/4] 完成：%s（%s 页，%.1f MB）" % (output, pages, size_mb))


def pdf_pages(pdf: Path) -> str:
    try:
        import pymupdf
        with pymupdf.open(pdf) as doc:
            return str(doc.page_count)
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
