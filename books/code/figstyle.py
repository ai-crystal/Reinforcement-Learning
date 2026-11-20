# -*- coding: utf-8 -*-
"""全书插图统一样式。所有绘图脚本 import 本模块。

约定：
  * 画布宽度用英寸，正文版心约 132 mm，常用 6.4 ~ 6.8 英寸
  * 图里不加标题，标题写在 Markdown 的图片说明里
  * 保存时直接输出 PNG 到 books/<册>/images/<NN>/，GitHub 与 PDF 共用一份
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CODE = Path(__file__).resolve().parent          # books/code
BOOKS = CODE.parent                             # books

# 与正文一致的色板：深绿为主，暖金点缀
DEEP = "#1E5C4F"
DEEP_L = "#4E8B7C"
DEEP_XL = "#CFE3DD"
DEEP_BG = "#EEF4F2"
GOLD = "#B08344"
GOLD_L = "#E5D2AE"
INK = "#191A1C"
GRAY = "#6E7178"
LGRAY = "#D5D8DE"
BG = "#F5F6F7"
WHITE = "#FFFFFF"
GREEN = "#4F7A5B"
GREEN_L = "#DCE8DE"
RED = "#9B4A3F"
RED_L = "#F0DEDA"
TEAL = "#2F6F73"
PLUM = "#6A4A6E"
PLUM_L = "#E7DDE8"

PALETTE = [DEEP, GOLD, TEAL, RED, GREEN, DEEP_L, PLUM]


def _pick_font() -> str:
    candidates = [
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("Noto Sans SC", "C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name, path in candidates:
        if name in available:
            return name
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
                return font_manager.FontProperties(fname=path).get_name()
            except Exception:
                continue
    return "DejaVu Sans"


FONT = _pick_font()


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "axes.grid": True,
        "grid.color": LGRAY,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "lines.linewidth": 1.4,
    })


def canvas(w: float = 6.4, h: float = 3.4):
    """带坐标轴的图，适合曲线与柱状图。"""
    setup()
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.96, bottom=0.15)
    return fig, ax


def diagram(w: float = 6.4, h: float = 3.0, xlim=(0, 100), ylim=(0, 50)):
    """示意图画布，坐标系用 0-100 百分比，方便按比例摆方块与箭头。"""
    setup()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.grid(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    return fig, ax


# 示意图按 6.8 in 宽绘制，排版时会缩到 132 mm，文字因此放大一档
FS = 1.32


def box(ax, x, y, w, h, text="", fc=BG, ec=DEEP_L, tc=INK, fs=8.4,
        lw=0.8, rounded=1.2, weight="normal", zorder=3, alpha=1.0,
        ha="center", va="center", ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=%s" % rounded,
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
                       zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else x + 2.0
        ty = y + h / 2 if va == "center" else y + h - 2.2
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs * FS, color=tc,
                zorder=zorder + 1, weight=weight, linespacing=1.35)
    return p


def label(ax, x, y, text, fs=8.2, color=INK, ha="center", va="center",
          weight="normal", zorder=6, style="normal", rotation=0):
    return ax.text(x, y, text, fontsize=fs * FS, color=color, ha=ha, va=va,
                   weight=weight, zorder=zorder, style=style, rotation=rotation,
                   linespacing=1.35)


def arrow(ax, p0, p1, color=DEEP, lw=1.0, style="-|>", rad=0.0, ls="-",
          zorder=2, shrinkA=1.5, shrinkB=1.5, mutation=9):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=mutation,
                        color=color, linewidth=lw, linestyle=ls,
                        connectionstyle="arc3,rad=%s" % rad,
                        shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
    ax.add_patch(a)
    return a


def line(ax, p0, p1, color=GRAY, lw=0.8, ls="-", zorder=1, alpha=1.0):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw,
            linestyle=ls, zorder=zorder, alpha=alpha, solid_capstyle="round")


def rect(ax, x, y, w, h, fc="none", ec=GRAY, lw=0.7, ls="-", zorder=1, alpha=1.0):
    p = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
                  linestyle=ls, zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    return p


def save(fig, book: str, nn: str, name: str) -> None:
    """保存到 books/<book>/images/<nn>/<name>。"""
    out_dir = BOOKS / book / "images" / nn
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print("saved", out.relative_to(BOOKS))
