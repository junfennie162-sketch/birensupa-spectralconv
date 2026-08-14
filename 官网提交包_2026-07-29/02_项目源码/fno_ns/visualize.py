#!/usr/bin/env python3
"""Save FNO prediction vs ground-truth vorticity figures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DEMO_MEDIA_DIR = Path(__file__).resolve().parents[1] / "demo" / "media"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
DEMO_BATCH = Path(__file__).resolve().parent / "checkpoints" / "demo_batch.pt"
DEMO_BATCH_META = Path(__file__).resolve().parent / "checkpoints" / "demo_batch_meta.json"


def _load_meta() -> dict:
    if not DEMO_BATCH_META.exists():
        return {}
    return json.loads(DEMO_BATCH_META.read_text())


def _sample_rel_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(prediction - truth) / denom)


def _save_main_panel(
    *,
    last_input: np.ndarray | None,
    truth: np.ndarray,
    prediction: np.ndarray,
    abs_error: np.ndarray,
    rel_error_map: np.ndarray,
    data_name: str,
    sample_index: int,
    target_time_step,
    sample_relative_l2: float,
    stamp: str,
) -> tuple[Path, Path]:
    shared_vmax = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1.0e-12))
    shared_vmin = -shared_vmax

    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    if last_input is None:
        axes[0, 0].set_title("Input N/A")
        axes[0, 0].axis("off")
    else:
        input_vmax = float(max(np.max(np.abs(last_input)), 1.0e-12))
        mesh = axes[0, 0].imshow(
            last_input,
            cmap="RdBu_r",
            origin="lower",
            vmin=-input_vmax,
            vmax=input_vmax,
        )
        axes[0, 0].set_title("Input (last frame)")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        figure.colorbar(mesh, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for axis, data, title, cmap, vmin, vmax in (
        (axes[0, 1], truth, "Ground truth", "RdBu_r", shared_vmin, shared_vmax),
        (axes[0, 2], prediction, "FNO prediction (SUPA)", "RdBu_r", shared_vmin, shared_vmax),
        (axes[1, 0], abs_error, "|Error| (abs pred-gt)", "magma", None, None),
        (axes[1, 1], rel_error_map, "Relative |error|/|GT|", "magma", None, None),
    ):
        kwargs = {"cmap": cmap, "origin": "lower"}
        if vmin is not None:
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        mesh = axis.imshow(data, **kwargs)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05,
        0.55,
        (
            f"data: {data_name}\n"
            f"sample: {sample_index}\n"
            f"target_t: {target_time_step}\n"
            f"sample_rel_L2: {sample_relative_l2:.6g}\n"
            f"shared Pred/GT colorbar: yes\n"
            f"abs + relative error maps: yes"
        ),
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
    )

    figure.suptitle(
        (
            "FNO-NS vorticity — pred vs ground truth\n"
            f"data={data_name}  sample={sample_index}  "
            f"target_t={target_time_step}  sample_rel_L2={sample_relative_l2:.6g}"
        ),
        fontsize=11,
    )
    figure.tight_layout()
    out_path = FIGURES_DIR / f"fno_ns_pred_vs_gt_{stamp}.png"
    demo_copy = DEMO_MEDIA_DIR / out_path.name
    figure.savefig(out_path, dpi=140)
    figure.savefig(demo_copy, dpi=140)
    plt.close(figure)
    return out_path, demo_copy


def _save_strip(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    data_name: str,
    target_time_step,
    stamp: str,
) -> tuple[Path, Path, list[dict]]:
    """Best / median / worst samples by per-sample relative L2."""
    batch = int(target.shape[0])
    scores = []
    for index in range(batch):
        truth = target[index, 0].numpy()
        prediction = pred[index, 0].numpy()
        scores.append((index, _sample_rel_l2(prediction, truth)))
    scores_sorted = sorted(scores, key=lambda item: item[1])
    picks = []
    if batch >= 1:
        picks.append(("best", scores_sorted[0]))
    if batch >= 3:
        picks.append(("median", scores_sorted[batch // 2]))
        picks.append(("worst", scores_sorted[-1]))
    elif batch == 2:
        picks.append(("worst", scores_sorted[-1]))

    figure, axes = plt.subplots(len(picks), 3, figsize=(11, 3.2 * max(len(picks), 1)))
    if len(picks) == 1:
        axes = np.array([axes])

    records = []
    for row, (label, (index, rel_l2)) in enumerate(picks):
        truth = target[index, 0].numpy()
        prediction = pred[index, 0].numpy()
        abs_error = np.abs(prediction - truth)
        shared_vmax = float(max(np.max(np.abs(truth)), np.max(np.abs(prediction)), 1.0e-12))
        for col, (data, title, cmap, use_shared) in enumerate(
            (
                (truth, f"{label} GT (#{index})", "RdBu_r", True),
                (prediction, f"{label} Pred", "RdBu_r", True),
                (abs_error, f"|err| relL2={rel_l2:.4g}", "magma", False),
            )
        ):
            kwargs = {"cmap": cmap, "origin": "lower"}
            if use_shared:
                kwargs["vmin"] = -shared_vmax
                kwargs["vmax"] = shared_vmax
            mesh = axes[row, col].imshow(data, **kwargs)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            figure.colorbar(mesh, ax=axes[row, col], fraction=0.046, pad=0.04)
        records.append({"label": label, "sample_index": index, "sample_relative_l2": rel_l2})

    figure.suptitle(
        f"FNO-NS sample strip — data={data_name} target_t={target_time_step}",
        fontsize=11,
    )
    figure.tight_layout()
    out_path = FIGURES_DIR / f"fno_ns_sample_strip_{stamp}.png"
    demo_copy = DEMO_MEDIA_DIR / out_path.name
    figure.savefig(out_path, dpi=140)
    figure.savefig(demo_copy, dpi=140)
    plt.close(figure)
    return out_path, demo_copy, records


def main() -> None:
    if not DEMO_BATCH.exists():
        raise SystemExit(f"missing {DEMO_BATCH}; run test_forward.py first")

    payload = torch.load(DEMO_BATCH, map_location="cpu", weights_only=False)
    meta = _load_meta()
    target = payload["target"]  # [B, 1, H, W]
    pred = payload["pred"]
    sample_index = int(meta.get("sample_index", 0))
    truth = target[sample_index, 0].numpy()
    prediction = pred[sample_index, 0].numpy()
    error = prediction - truth
    abs_error = np.abs(error)
    rel_error_map = abs_error / np.maximum(np.abs(truth), 1.0e-6)

    sample_relative_l2 = meta.get("sample_relative_l2")
    if sample_relative_l2 is None:
        sample_relative_l2 = _sample_rel_l2(prediction, truth)

    data_name = meta.get("data", "unknown")
    target_time_step = meta.get("target_time_step", "?")
    last_input = None
    if "input" in payload:
        last_input = payload["input"][sample_index, -1].numpy()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")

    out_path, demo_copy = _save_main_panel(
        last_input=last_input,
        truth=truth,
        prediction=prediction,
        abs_error=abs_error,
        rel_error_map=rel_error_map,
        data_name=data_name,
        sample_index=sample_index,
        target_time_step=target_time_step,
        sample_relative_l2=sample_relative_l2,
        stamp=stamp,
    )
    strip_path, strip_demo, strip_records = _save_strip(
        target=target,
        pred=pred,
        data_name=data_name,
        target_time_step=target_time_step,
        stamp=stamp,
    )

    print(
        {
            "figure": str(out_path),
            "demo_copy": str(demo_copy),
            "strip_figure": str(strip_path),
            "strip_demo_copy": str(strip_demo),
            "sample_relative_l2": sample_relative_l2,
            "strip": strip_records,
            "data": data_name,
        }
    )

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        fno = summary.setdefault("fno_ns", {})
        figures = list(fno.get("figures") or [])
        for path in (out_path, strip_path):
            rel = str(path.relative_to(RESULTS_DIR.parent))
            if rel not in figures:
                figures.append(rel)
        fno["figures"] = figures
        fno["visualization"] = {
            "shared_colorbar_for_pred_gt": True,
            "absolute_and_relative_error_maps": True,
            "sample_index": sample_index,
            "target_time_step": target_time_step,
            "sample_relative_l2": sample_relative_l2,
            "data": data_name,
            "figure": str(out_path.relative_to(RESULTS_DIR.parent)),
            "demo_copy": str(demo_copy.relative_to(RESULTS_DIR.parent)),
            "sample_strip": str(strip_path.relative_to(RESULTS_DIR.parent)),
            "sample_strip_records": strip_records,
        }
        summary.setdefault("meta", {})["updated_at"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
