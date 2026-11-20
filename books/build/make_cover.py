# -*- coding: utf-8 -*-
"""从背景图生成封面与封底的视觉底图（不含文字，文字由 LaTeX 模板排版）。

用法：
    python books/build/make_cover.py books/concepts E:/Study/0-logo/logo3.png
    python books/build/make_cover.py books/concepts E:/Study/0-logo/logo3.png --anchor 0.4

输出：
    books/<册>/images/cover/cover-art.jpg
    books/<册>/images/cover/back-art.jpg

封面底图按页面宽度铺满、取上部若干比例，下缘渐变融入底色，保证书名始终读得清。
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# 页面 170 x 240 mm；封面图占上部 150 mm，其余交还给模板底色
PAGE_RATIO = 170 / 240.0
ART_RATIO = 170 / 150.0
BG = (7, 16, 14)


def load_crop(path: Path, ratio: float, anchor: float) -> Image.Image:
    im = Image.open(path).convert("RGB")
    target_h = int(round(im.width / ratio))
    if target_h <= im.height:
        top = int((im.height - target_h) * anchor)
        top = max(0, min(top, im.height - target_h))
        box = (0, top, im.width, top + target_h)
    else:
        target_w = int(round(im.height * ratio))
        left = (im.width - target_w) // 2
        box = (left, 0, left + target_w, im.height)
    return im.crop(box)


def fade_to_bg(im: Image.Image, start: float = 0.55, power: float = 1.5) -> Image.Image:
    w, h = im.size
    mask = Image.new("L", (1, h), 0)
    px = mask.load()
    top = int(h * start)
    for y in range(h):
        if y < top:
            v = 0
        else:
            t = (y - top) / float(h - top)
            v = int(255 * (t ** power))
        px[0, y] = v
    mask = mask.resize((w, h))
    base = Image.new("RGB", (w, h), BG)
    return Image.composite(base, im, mask)


def grade(im: Image.Image, dim: float = 0.80, tint=(8, 20, 26), amount: float = 0.20) -> Image.Image:
    im = im.point(lambda v: int(v * dim))
    return Image.blend(im, Image.new("RGB", im.size, tint), amount)


def make_cover(bg: Path, out: Path, anchor: float, width: int = 1700) -> None:
    im = load_crop(bg, ART_RATIO, anchor)
    im = grade(im)
    im = fade_to_bg(im)
    h = int(round(width / ART_RATIO))
    im = im.resize((width, h), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "JPEG", quality=92, optimize=True)
    print("cover ->", out, im.size)


def make_back(bg: Path, out: Path, anchor: float, width: int = 1700) -> None:
    """封底：同一张图做虚化与压暗，中部偏上稍亮，留出文字区。"""
    im = load_crop(bg, PAGE_RATIO, anchor)
    im = im.filter(ImageFilter.GaussianBlur(9))
    im = im.point(lambda v: int(v * 0.62))
    im = Image.blend(im, Image.new("RGB", im.size, BG), 0.35)
    w, h = im.size
    mask = Image.new("L", (1, h), 0)
    px = mask.load()
    for y in range(h):
        t = y / float(h - 1)
        px[0, y] = int(150 + 70 * (1 - abs(t - 0.42) * 2))
    mask = mask.resize((w, h))
    im = Image.composite(Image.new("RGB", (w, h), (4, 10, 9)), im, mask)
    im = im.resize((width, int(round(width / PAGE_RATIO))), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "JPEG", quality=92, optimize=True)
    print("back  ->", out, im.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("background")
    ap.add_argument("--anchor", type=float, default=0.35,
                    help="裁切时纵向取景位置，0 取上缘，1 取下缘")
    ap.add_argument("--back-anchor", type=float, default=0.55)
    args = ap.parse_args()

    book = Path(args.book)
    bg = Path(args.background)
    make_cover(bg, book / "images" / "cover" / "cover-art.jpg", args.anchor)
    make_back(bg, book / "images" / "cover" / "back-art.jpg", args.back_anchor)


if __name__ == "__main__":
    main()
