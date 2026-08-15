#!/usr/bin/env python3
"""Draw contest flow-field figures from official NS64 demo_batch.pt.

Two languages, two pictures each:
  1) typical sample — last input / ground truth / prediction
  2) best / typical / worst — ground truth / prediction / |error|

Typical = the test sample whose relative L2 is closest to the official
mean 0.035012. Relative-error heatmaps are not drawn (near-zero GT
makes them look noisy and is not the contest metric).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DEMO_MEDIA_DIR = Path(__file__).resolve().parents[1] / "demo" / "media"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DEMO_BATCH = Path(__file__).resolve().parent / "checkpoints" / "demo_batch.pt"
DEMO_BATCH_META = Path(__file__).resolve().parent / "checkpoints" / "demo_batch_meta.json"
OFFICIAL_L2 = 0.035012

TEXT = {
    "zh": {
        "font_candidates": (
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "Source Han Sans SC",
            "Droid Sans Fallback",
            "AR PL UMing CN",
            "SimHei",
        ),
        "hero_title": "典型样本：输入末帧 / 真值 / 预测",
        "hero_sub": (
            "官方公开 NS64 · 前 10 帧预测第 11 帧\n"
            "正式成绩是全部 128 条测试的平均相对 L2 = {official:.6f}\n"
            "本图选最接近该均值的一条（本条 {sample:.4f}）· 红蓝表示涡度"
        ),
        "input": "输入（第 10 帧）",
        "gt": "真值（第 11 帧）",
        "pred": "本模型预测",
        "err": "绝对误差 |预测−真值|",
        "strip_title": "最好 / 典型 / 最差 对照",
        "strip_sub": (
            "同一官方测试集。左=真值，中=预测，右=绝对误差。\n"
            "典型 = 相对 L2 最接近正式成绩 {official:.6f} 的样本"
        ),
        "best": "最好",
        "typical": "典型（接近正式成绩）",
        "worst": "最差",
        "rel": "相对 L2",
        "hero_name": "01_典型样本_预测与真值.png",
        "strip_name": "02_最好_典型_最差.png",
    },
    "en": {
        "font_candidates": ("DejaVu Sans", "Noto Sans", "Liberation Sans"),
        "hero_title": "Typical sample: last input / ground truth / prediction",
        "hero_sub": (
            "Official public NS64 · frames 1–10 → frame 11\n"
            "Contest score = mean relative L2 over 128 test samples = {official:.6f}\n"
            "This panel is the sample closest to that mean (this sample {sample:.4f}). Red/blue = vorticity."
        ),
        "input": "Input (frame 10)",
        "gt": "Ground truth (frame 11)",
        "pred": "Our prediction",
        "err": "Absolute error |pred−GT|",
        "strip_title": "Best / typical / worst",
        "strip_sub": (
            "Same official test split. Left=GT, middle=pred, right=absolute error.\n"
            "Typical = sample whose relative L2 is closest to the contest score {official:.6f}"
        ),
        "best": "Best",
        "typical": "Typical (near contest score)",
        "worst": "Worst",
        "rel": "rel. L2",
        "hero_name": "01_typical_sample_pred_vs_gt.png",
        "strip_name": "02_best_typical_worst.png",
    },
}


def _load_meta() -> dict:
    if not DEMO_BATCH_META.exists():
        return {}
    return json.loads(DEMO_BATCH_META.read_text())


def _sample_rel_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(prediction - truth) / denom)


def _pick_font(lang: str) -> str | None:
    wanted = TEXT[lang]["font_candidates"]
    available = {item.name: item.fname for item in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in available:
            return name
    if lang != "zh":
        return None
    for item in font_manager.fontManager.ttflist:
        path = item.fname.lower()
        if "cjk" in path or "wqy" in path or "uming" in path or "noto-sans-cjk" in path:
            return item.name
    return None


_CJK_FILES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
) + tuple(
    str(p)
    for p in sorted((Path(__file__).resolve().parent / "assets").glob("*"))
    if p.suffix.lower() in {".ttf", ".otf", ".ttc"}
)


def _apply_font(lang: str) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    name = _pick_font(lang)
    path = None
    if name:
        by_name = {item.name: item.fname for item in font_manager.fontManager.ttflist}
        path = by_name.get(name)
    elif lang == "zh":
        for candidate in _CJK_FILES:
            if Path(candidate).exists():
                font_manager.fontManager.addfont(candidate)
                fp = font_manager.FontProperties(fname=candidate)
                name = fp.get_name()
                path = candidate
                break
    if not name:
        return
    # DejaVu covers U+2212 minus; CJK fonts often do not.
    if lang == "zh":
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]
        print(f"zh_font={name!r} path={path or '?'}")
    else:
        plt.rcParams["font.family"] = name


def _imshow(axis, data, cmap, vmin=None, vmax=None):
    kwargs = {"cmap": cmap, "origin": "lower"}
    if vmin is not None:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    mesh = axis.imshow(data, **kwargs)
    axis.set_xticks([])
    axis.set_yticks([])
    return mesh


def _scores(target: torch.Tensor, pred: torch.Tensor) -> list[tuple[int, float]]:
    rows = []
    for index in range(int(target.shape[0])):
        rows.append(
            (
                index,
                _sample_rel_l2(pred[index, 0].numpy(), target[index, 0].numpy()),
            )
        )
    return rows


def _picks(scores: list[tuple[int, float]], official: float) -> dict[str, tuple[int, float]]:
    ordered = sorted(scores, key=lambda item: item[1])
    typical = min(scores, key=lambda item: abs(item[1] - official))
    return {
        "best": ordered[0],
        "typical": typical,
        "worst": ordered[-1],
    }


def _save_hero(
    *,
    lang: str,
    last_input: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    sample_index: int,
    sample_l2: float,
    official: float,
) -> tuple[Path, Path]:
    t = TEXT[lang]
    _apply_font(lang)
    shared = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1.0e-12))
    input_v = float(max(np.max(np.abs(last_input)), 1.0e-12))

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    panels = (
        (last_input, t["input"], "RdBu_r", -input_v, input_v),
        (truth, t["gt"], "RdBu_r", -shared, shared),
        (prediction, t["pred"], "RdBu_r", -shared, shared),
    )
    for axis, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        mesh = _imshow(axis, data, cmap, vmin, vmax)
        axis.set_title(title, fontsize=12, pad=8)
        figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(t["hero_title"], fontsize=14, fontweight="bold", y=1.02)
    figure.text(
        0.5,
        -0.02,
        t["hero_sub"].format(official=official, sample=sample_l2),
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout()
    out_path = FIGURES_DIR / t["hero_name"]
    demo_copy = DEMO_MEDIA_DIR / t["hero_name"]
    figure.savefig(out_path, dpi=160, bbox_inches="tight")
    figure.savefig(demo_copy, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return out_path, demo_copy


def _save_strip(
    *,
    lang: str,
    target: torch.Tensor,
    pred: torch.Tensor,
    picks: dict[str, tuple[int, float]],
    official: float,
) -> tuple[Path, Path]:
    t = TEXT[lang]
    _apply_font(lang)
    order = ("best", "typical", "worst")
    figure, axes = plt.subplots(3, 3, figsize=(12.4, 11.2))
    for row, key in enumerate(order):
        index, rel_l2 = picks[key]
        truth = target[index, 0].numpy()
        prediction = pred[index, 0].numpy()
        abs_error = np.abs(prediction - truth)
        shared = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1.0e-12))
        label = t[key]
        panels = (
            (truth, f"{label} · {t['gt']}", "RdBu_r", -shared, shared),
            (prediction, f"{label} · {t['pred']}", "RdBu_r", -shared, shared),
            (abs_error, f"{t['err']}\n{t['rel']} = {rel_l2:.4f}", "magma", 0.0, None),
        )
        for col, (data, title, cmap, vmin, vmax) in enumerate(panels):
            mesh = _imshow(axes[row, col], data, cmap, vmin, vmax)
            axes[row, col].set_title(title, fontsize=11, pad=6)
            figure.colorbar(mesh, ax=axes[row, col], fraction=0.046, pad=0.04)

    figure.suptitle(t["strip_title"], fontsize=14, fontweight="bold", y=0.995)
    figure.text(
        0.5,
        0.005,
        t["strip_sub"].format(official=official),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path = FIGURES_DIR / t["strip_name"]
    demo_copy = DEMO_MEDIA_DIR / t["strip_name"]
    figure.savefig(out_path, dpi=160, bbox_inches="tight")
    figure.savefig(demo_copy, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return out_path, demo_copy


def main() -> None:
    if not DEMO_BATCH.exists():
        raise SystemExit(f"missing {DEMO_BATCH}; run render_official_demo.py first")

    payload = torch.load(DEMO_BATCH, map_location="cpu", weights_only=False)
    meta = _load_meta()
    target = payload["target"]
    pred = payload["pred"]
    inputs = payload["input"]
    official = float(meta.get("official_relative_l2", OFFICIAL_L2))
    scores = _scores(target, pred)
    picks = _picks(scores, official)
    typical_index, typical_l2 = picks["typical"]
    last_input = inputs[typical_index, -1].numpy()
    truth = target[typical_index, 0].numpy()
    prediction = pred[typical_index, 0].numpy()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    drawn = {}
    for lang in ("zh", "en"):
        hero, hero_demo = _save_hero(
            lang=lang,
            last_input=last_input,
            truth=truth,
            prediction=prediction,
            sample_index=typical_index,
            sample_l2=typical_l2,
            official=official,
        )
        strip, strip_demo = _save_strip(
            lang=lang,
            target=target,
            pred=pred,
            picks=picks,
            official=official,
        )
        drawn[lang] = {
            "hero": str(hero),
            "hero_demo": str(hero_demo),
            "strip": str(strip),
            "strip_demo": str(strip_demo),
        }

    print(
        {
            "typical_index": typical_index,
            "typical_rel_l2": typical_l2,
            "official_relative_l2": official,
            "best": picks["best"],
            "worst": picks["worst"],
            "files": drawn,
        }
    )

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        fno = summary.setdefault("fno_ns", {})
        fno["visualization"] = {
            "data": meta.get("data", "public_ns64"),
            "layout": "typical_triplet + best_typical_worst_strip",
            "sample_index": typical_index,
            "sample_relative_l2": typical_l2,
            "official_relative_l2": official,
            "target_time_step": meta.get("target_time_step", 10),
            "picks": {
                key: {"sample_index": idx, "sample_relative_l2": val}
                for key, (idx, val) in picks.items()
            },
            "zh": drawn["zh"],
            "en": drawn["en"],
        }
        summary.setdefault("meta", {})["updated_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
