#!/usr/bin/env python3
"""Offline Error Autopsy D (epochs=0): freeze_r9 vs freeze_r10 read-only diagnostics.

Does NOT train, promote, overwrite summary.json, or run test_perf.
Writes JSON + figures + VERDICT under --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT.parent
sys.path.insert(0, str(ROOT))

from dataset import SequenceVorticityDataset, load_or_build_ns_like, split_train_test
from model import FNO2d
from train_public_ns64_boost import evaluate, predict, relative_l2

SPECTRUM_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 24), (24, 32)]
BOOTSTRAP_B = 2000
WORST_K = 16


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_indices(n_total: int, n_train: int, n_test: int, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    train_idx = perm[:n_train].tolist()
    test_idx = perm[n_train : n_train + n_test].tolist()
    return train_idx, test_idx


def load_model(ckpt_path: Path) -> tuple[FNO2d, dict]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    modes = int(blob.get("modes", 16))
    width = int(blob.get("width", 32))
    residual = bool(blob.get("residual", True))
    model = FNO2d(modes1=modes, modes2=modes, width=width, n_layers=4, in_channels=10, out_channels=1)
    model.load_state_dict(blob["model"])
    model.eval()
    meta = {
        "path": str(ckpt_path),
        "modes": modes,
        "width": width,
        "residual": residual,
        "test_l2_ckpt": blob.get("test_l2"),
        "epoch": blob.get("epoch"),
        "promoted_tag": blob.get("promoted_tag"),
        "data_source": blob.get("data_source"),
    }
    return model, meta


def sample_rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return relative_l2(pred, target)


def mean_of_batch_means(scores: list[float], batch_size: int = 16) -> float:
    if not scores:
        return float("nan")
    batch_means = []
    for i in range(0, len(scores), batch_size):
        chunk = scores[i : i + batch_size]
        batch_means.append(sum(chunk) / len(chunk))
    return sum(batch_means) / len(batch_means)


def paired_bootstrap_ci(
    values: np.ndarray, n_boot: int = BOOTSTRAP_B, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for the mean of values via paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    if den < 1e-12:
        return 0.0
    return float((rx * ry).sum() / den)


def radial_masks(h: int, w: int, device: torch.device) -> list[torch.Tensor]:
    fy = torch.fft.fftfreq(h, d=1.0, device=device) * h
    fx = torch.fft.fftfreq(w, d=1.0, device=device) * w
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(yy * yy + xx * xx)
    masks = []
    for lo, hi in SPECTRUM_BINS:
        masks.append((rr >= lo) & (rr < hi))
    return masks


def spectrum_contrib(err: torch.Tensor, gt: torch.Tensor, masks: list[torch.Tensor]) -> dict:
    """err/gt: [H,W]. Returns C_b and R_b per bin."""
    e_hat = torch.fft.fft2(err)
    g_hat = torch.fft.fft2(gt)
    pe = (e_hat.real**2 + e_hat.imag**2)
    pg = (g_hat.real**2 + g_hat.imag**2)
    total_e = float(pe.sum().clamp_min(1e-12).item())
    out_c, out_r = {}, {}
    for (lo, hi), m in zip(SPECTRUM_BINS, masks):
        key = f"[{lo},{hi})"
        e_b = float(pe[m].sum().item())
        g_b = float(pg[m].sum().item())
        out_c[key] = e_b / total_e
        out_r[key] = e_b / (g_b + 1e-12)
    return {"C_b": out_c, "R_b": out_r}


def dynamics_features(traj: torch.Tensor) -> dict:
    """traj: [T,H,W] full sample. Features anchored at frames 9/10 (0-index)."""
    u10 = traj[9]
    u11 = traj[10]
    qt = float((torch.norm(u11 - u10) / torch.norm(u11).clamp_min(1e-12)).item())
    enstrophy = float((u11 * u11).sum().item())
    # central-diff gradient energy (periodic)
    dx = torch.roll(u11, -1, dims=-1) - torch.roll(u11, 1, dims=-1)
    dy = torch.roll(u11, -1, dims=-2) - torch.roll(u11, 1, dims=-2)
    grad_e = float((dx * dx + dy * dy).sum().item())
    # high-freq energy ratio via rfft2 magnitude outside modes<12
    f = torch.fft.rfft2(u11)
    mag2 = f.real**2 + f.imag**2
    h, wf = mag2.shape
    yy = torch.arange(h, dtype=mag2.dtype)[:, None]
    xx = torch.arange(wf, dtype=mag2.dtype)[None, :]
    yy = torch.minimum(yy, h - yy)
    rr = torch.sqrt(yy**2 + xx**2)
    total = float(mag2.sum().clamp_min(1e-12).item())
    hf = float(mag2[rr >= 12].sum().item()) / total
    return {
        "q_t": qt,
        "enstrophy": enstrophy,
        "grad_energy": grad_e,
        "hf_energy_ratio": hf,
        "vmax": float(u11.max().item()),
        "vmin": float(u11.min().item()),
    }


@torch.no_grad()
def diagnose_one(
    model: FNO2d,
    traj: torch.Tensor,
    residual: bool,
    masks: list[torch.Tensor],
) -> dict:
    """traj: [T,H,W], T>=12. Returns metrics for one sample."""
    t_in = 10
    # e1: frames 0..9 -> predict frame 10
    x0 = traj[:t_in].unsqueeze(0)  # [1,10,H,W]
    y1 = traj[t_in : t_in + 1].unsqueeze(0)
    pred1 = predict(model, x0, residual)
    e1 = sample_rel_l2(pred1, y1)

    # e2_TF: frames 1..10 (GT) -> predict frame 11
    x_tf = traj[1 : t_in + 1].unsqueeze(0)
    y2 = traj[t_in + 1 : t_in + 2].unsqueeze(0)
    pred2_tf = predict(model, x_tf, residual)
    e2_tf = sample_rel_l2(pred2_tf, y2)

    # e2_AR: frames 1..9 + pred1 -> predict frame 11
    x_ar = torch.cat([traj[1:t_in].unsqueeze(0), pred1], dim=1)
    pred2_ar = predict(model, x_ar, residual)
    e2_ar = sample_rel_l2(pred2_ar, y2)

    g = e2_ar - e2_tf

    # rollout from clean pred1 through remaining frames (up to T-1)
    T = traj.shape[0]
    cur = x0.clone()
    rollout = [e1]
    next_pred = pred1
    for t in range(t_in + 1, T):
        cur = torch.cat([cur[:, 1:], next_pred], dim=1)
        next_pred = predict(model, cur, residual)
        target = traj[t : t + 1].unsqueeze(0)
        rollout.append(sample_rel_l2(next_pred, target))

    err = (pred1 - y1).squeeze(0).squeeze(0)
    gt = y1.squeeze(0).squeeze(0)
    spec = spectrum_contrib(err, gt, masks)
    feats = dynamics_features(traj)

    return {
        "e1": e1,
        "e2_TF": e2_tf,
        "e2_AR": e2_ar,
        "g": g,
        "rollout_curve": rollout,
        "spectrum": spec,
        "features": feats,
        "pred1": pred1.squeeze(0).squeeze(0).cpu(),
        "gt1": gt.cpu(),
        "err1": err.cpu(),
    }


def group_mean_spectrum(rows: list[dict], indices: list[int]) -> dict:
    keys = [f"[{lo},{hi})" for lo, hi in SPECTRUM_BINS]
    c_acc = {k: 0.0 for k in keys}
    r_acc = {k: 0.0 for k in keys}
    n = max(len(indices), 1)
    for i in indices:
        sp = rows[i]["spectrum"]
        for k in keys:
            c_acc[k] += sp["C_b"][k]
            r_acc[k] += sp["R_b"][k]
    return {
        "C_b": {k: c_acc[k] / n for k in keys},
        "R_b": {k: r_acc[k] / n for k in keys},
        "n": len(indices),
    }


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def plot_protocol(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    table = [
        ["Item", "FNO paper 0.0128", "Our freeze_r9"],
        ["Task", "Recurrent traj to T=50", "Clean 10→1 single step"],
        ["Data length", "T=50 snapshots", "T=20 file (HF N1200)"],
        ["Test N", "200", "128 (seed=20260722)"],
        ["Modes / width", "12 / 32", "16 / 32"],
        ["Metric meaning", "Rollout traj aggregate", "Per-sample rel-L2 mean"],
        ["Comparable?", "—", "NO — not isomorphic"],
    ]
    tbl = ax.table(cellText=table, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.45)
    for j in range(3):
        tbl[(0, j)].set_facecolor("#d9e2ef")
    ax.set_title("Why paper 0.0128 is not our single-step target", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_error_decomp(rows: list[dict], out: Path) -> None:
    e1 = [r["e1"] for r in rows]
    e2tf = [r["e2_TF"] for r in rows]
    e2ar = [r["e2_AR"] for r in rows]
    g = [r["g"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot([e1, e2tf, e2ar], tick_labels=["e1", "e2_TF", "e2_AR"], showfliers=True)
    axes[0].set_ylabel("relative L2")
    axes[0].set_title("Clean / TF / AR errors (freeze_r9)")
    axes[1].hist(g, bins=24, color="#4c72b0", edgecolor="white")
    axes[1].axvline(0.0, color="crimson", ls="--", lw=1)
    axes[1].set_xlabel("g = e2_AR − e2_TF")
    axes[1].set_title("Exposure gap distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_spectrum_strip(
    rows: list[dict],
    order: list[int],
    out: Path,
) -> None:
    n = len(order)
    best = order[:WORST_K]
    worst = order[-WORST_K:]
    mid = order[n // 2 - WORST_K // 2 : n // 2 - WORST_K // 2 + WORST_K]
    groups = {
        "all": list(range(n)),
        "best16": best,
        "median16": mid,
        "worst16": worst,
    }
    keys = [f"[{lo},{hi})" for lo, hi in SPECTRUM_BINS]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, idxs) in zip(axes.ravel(), groups.items()):
        gspec = group_mean_spectrum(rows, idxs)
        vals = [gspec["C_b"][k] for k in keys]
        ax.bar(range(len(keys)), vals, color="#55a868")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, max(0.01, max(vals) * 1.15))
        ax.set_title(f"{name}: error energy C_b")
        ax.set_ylabel("fraction of ‖err̂‖²")
    # bottom: worst sample error map strip
    fig.suptitle("Radial spectrum error contribution (freeze_r9)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # companion heatmaps for best/median/worst single samples
    fig2, axes2 = plt.subplots(3, 3, figsize=(9, 8))
    picks = [
        ("best", order[0]),
        ("median", order[n // 2]),
        ("worst", order[-1]),
    ]
    for row_i, (label, idx) in enumerate(picks):
        r = rows[idx]
        for col_i, (title, arr) in enumerate(
            [("GT", r["gt1"]), ("Pred", r["pred1"]), ("Error", r["err1"])]
        ):
            im = axes2[row_i, col_i].imshow(arr.numpy(), cmap="RdBu_r")
            axes2[row_i, col_i].set_title(f"{label}#{idx} {title}\ne1={r['e1']:.4f}", fontsize=8)
            axes2[row_i, col_i].axis("off")
            fig2.colorbar(im, ax=axes2[row_i, col_i], fraction=0.046)
    fig2.suptitle("Vorticity GT / Pred / Error strip", y=1.01)
    fig2.tight_layout()
    heat_out = out.with_name(out.stem + "_heatmaps.png")
    fig2.savefig(heat_out, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    return heat_out


def mirror_figures(out_dir: Path, names: list[str]) -> None:
    fig_dir = SUBMISSION / "results" / "figures"
    media = SUBMISSION / "demo" / "media"
    archive = media / "archive_history"
    fig_dir.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = out_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, fig_dir / name)
        dst = media / name
        if dst.exists():
            # keep dated autopsy figures; do not archive pred_vs_gt primary unless colliding
            pass
        shutil.copy2(src, dst)


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SUBMISSION / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data, source = load_or_build_ns_like(
        n_samples=args.n_train + args.n_test,
        resolution=64,
        n_times=20,
        seed=args.seed,
        version="v2",
    )
    if not str(source).startswith("file:navier_stokes"):
        raise SystemExit(f"need public NS64 file, got {source}")

    data_path = ROOT / "data" / str(source).split("file:", 1)[-1]
    data_sha = sha256_file(data_path) if data_path.exists() else "MISSING"

    train_idx, test_idx = split_indices(data.shape[0], args.n_train, args.n_test, args.seed)
    _, test_data = split_train_test(data, args.n_train, args.n_test, seed=args.seed)

    ckpt_a = Path(args.ckpt_a)
    ckpt_b = Path(args.ckpt_b)
    if not ckpt_a.is_absolute():
        ckpt_a = ROOT / ckpt_a
    if not ckpt_b.is_absolute():
        ckpt_b = ROOT / ckpt_b

    model_a, meta_a = load_model(ckpt_a)
    model_b, meta_b = load_model(ckpt_b)
    residual_a = meta_a["residual"]
    residual_b = meta_b["residual"]

    # D0: evaluator cross-check on ckpt A
    ds1 = SequenceVorticityDataset(test_data, 10, 1)
    loader = DataLoader(ds1, batch_size=16, shuffle=False)
    eval_batch_mean = evaluate(model_a, loader, residual_a)
    per_sample = []
    with torch.no_grad():
        for i in range(len(ds1)):
            x, y = ds1[i]
            pred = predict(model_a, x.unsqueeze(0), residual_a)
            per_sample.append(sample_rel_l2(pred, y.unsqueeze(0)))
    mean_128 = float(sum(per_sample) / len(per_sample))
    mean_batch_of_means = mean_of_batch_means(per_sample, 16)

    d0 = {
        "protocol": {
            "data_source": source,
            "data_path": str(data_path),
            "sha256": data_sha,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "seed": args.seed,
            "T_in": 10,
            "T_out_eval": 1,
            "task": "clean_10_to_1_single_step",
        },
        "test_indices": test_idx,
        "evaluator": {
            "relative_l2_def": "mean over spatial dims of ||pred-gt||/||gt||; then reduce",
            "evaluate_fn": "mean of per-batch means (batch already mean of samples)",
            "diagnose_primary": "mean of 128 per-sample relative_l2 (promote_public_ckpt style)",
            "ckpt_a_evaluate_batch_mean": eval_batch_mean,
            "ckpt_a_mean_128": mean_128,
            "ckpt_a_mean_of_batch_means_bs16": mean_batch_of_means,
            "ckpt_a_recorded_test_l2": meta_a.get("test_l2_ckpt"),
        },
        "sched_vs_pushforward": {
            "sched_multistep_loss": (
                "for each step: loss on pred vs GT; nxt = pred.detach() with prob p_ar "
                "or soft_alpha blend; multi-step losses accumulated"
            ),
            "true_pushforward": (
                "stopgrad step-1 prediction as input; supervise only step-2 "
                "(optionally with clean anchor) — not identical to sched"
            ),
            "note": "Existing sched already detaches preds but is not pure PF.",
        },
        "checkpoints": {"a": meta_a, "b": meta_b},
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(out_dir / "d0_protocol.json", d0)

    h, w = int(test_data.shape[-2]), int(test_data.shape[-1])
    masks = radial_masks(h, w, torch.device("cpu"))

    def run_ckpt(model: FNO2d, residual: bool, tag: str) -> list[dict]:
        rows = []
        for i in range(test_data.shape[0]):
            traj = test_data[i]  # [T,H,W]
            rec = diagnose_one(model, traj, residual, masks)
            rec["local_index"] = i
            rec["global_index"] = test_idx[i]
            # drop heavy tensors from JSON later
            rows.append(rec)
            if (i + 1) % 16 == 0:
                print(f"[{tag}] {i+1}/{test_data.shape[0]}", flush=True)
        return rows

    print("Diagnosing ckpt A (freeze_r9)...", flush=True)
    rows_a = run_ckpt(model_a, residual_a, "r9")
    print("Diagnosing ckpt B (freeze_r10)...", flush=True)
    rows_b = run_ckpt(model_b, residual_b, "r10")

    def serialize_rows(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            out.append(
                {
                    "local_index": r["local_index"],
                    "global_index": r["global_index"],
                    "e1": r["e1"],
                    "e2_TF": r["e2_TF"],
                    "e2_AR": r["e2_AR"],
                    "g": r["g"],
                    "rollout_curve": r["rollout_curve"],
                    "spectrum": r["spectrum"],
                    "features": r["features"],
                }
            )
        return out

    write_json(out_dir / "per_sample_r9.json", serialize_rows(rows_a))
    write_json(out_dir / "per_sample_r10.json", serialize_rows(rows_b))

    # D1 AR hypothesis
    e1 = np.array([r["e1"] for r in rows_a], dtype=np.float64)
    g = np.array([r["g"] for r in rows_a], dtype=np.float64)
    e2_tf = np.array([r["e2_TF"] for r in rows_a], dtype=np.float64)
    e2_ar = np.array([r["e2_AR"] for r in rows_a], dtype=np.float64)
    g_mean, g_lo, g_hi = paired_bootstrap_ci(g, seed=args.seed)
    rho = spearman_rho(e1, g)
    order_e1 = list(np.argsort(e1))
    order_g = list(np.argsort(-g))  # largest gap first
    worst_e1 = set(order_e1[-WORST_K:])
    worst_g = set(order_g[:WORST_K])
    overlap = len(worst_e1 & worst_g)
    cond1 = bool(np.median(g) > 0 and g_lo > 0)
    cond2 = bool(rho >= 0.4)
    cond3 = bool(overlap >= 8)
    ar_pass = cond1 and cond2 and cond3

    # half-split consistency on rho and median(g)
    half = len(e1) // 2
    rho_h1 = spearman_rho(e1[:half], g[:half])
    rho_h2 = spearman_rho(e1[half:], g[half:])
    med_h1 = float(np.median(g[:half]))
    med_h2 = float(np.median(g[half:]))
    half_conflict = bool((med_h1 > 0) != (med_h2 > 0)) or (
        abs(rho_h1 - rho_h2) > 0.5 and max(abs(rho_h1), abs(rho_h2)) >= 0.4
    )

    d1 = {
        "summary": {
            "mean_e1": float(e1.mean()),
            "median_e1": float(np.median(e1)),
            "mean_e2_TF": float(e2_tf.mean()),
            "mean_e2_AR": float(e2_ar.mean()),
            "mean_g": g_mean,
            "median_g": float(np.median(g)),
            "g_bootstrap_mean_ci95": [g_lo, g_hi],
            "spearman_rho_e1_g": rho,
            "worst16_e1_g_overlap": overlap,
        },
        "preregister_conditions": {
            "cond1_median_g_pos_and_ci_lo_pos": cond1,
            "cond2_rho_ge_0.4": cond2,
            "cond3_worst16_overlap_ge_8": cond3,
            "all_pass": ar_pass,
        },
        "half_split": {
            "rho_first64": rho_h1,
            "rho_last64": rho_h2,
            "median_g_first64": med_h1,
            "median_g_last64": med_h2,
            "conflict": half_conflict,
        },
        "mean_rollout_curve": [
            float(np.mean([r["rollout_curve"][t] for r in rows_a if t < len(r["rollout_curve"])]))
            for t in range(max(len(r["rollout_curve"]) for r in rows_a))
        ],
    }
    write_json(out_dir / "d1_ar_hypothesis.json", d1)

    # D2 spectrum
    mid = order_e1[len(order_e1) // 2 - WORST_K // 2 : len(order_e1) // 2 - WORST_K // 2 + WORST_K]
    d2 = {
        "bins": [f"[{lo},{hi})" for lo, hi in SPECTRUM_BINS],
        "freeze_r9": {
            "all": group_mean_spectrum(rows_a, list(range(len(rows_a)))),
            "best16": group_mean_spectrum(rows_a, order_e1[:WORST_K]),
            "median16": group_mean_spectrum(rows_a, mid),
            "worst16": group_mean_spectrum(rows_a, order_e1[-WORST_K:]),
        },
    }
    e1_b = np.array([r["e1"] for r in rows_b], dtype=np.float64)
    order_b = list(np.argsort(e1_b))
    mid_b = order_b[len(order_b) // 2 - WORST_K // 2 : len(order_b) // 2 - WORST_K // 2 + WORST_K]
    d2["freeze_r10"] = {
        "all": group_mean_spectrum(rows_b, list(range(len(rows_b)))),
        "best16": group_mean_spectrum(rows_b, order_b[:WORST_K]),
        "median16": group_mean_spectrum(rows_b, mid_b),
        "worst16": group_mean_spectrum(rows_b, order_b[-WORST_K:]),
    }
    # concentration: max C_b among mid/high bins for worst16
    worst_c = d2["freeze_r9"]["worst16"]["C_b"]
    max_bin = max(worst_c.items(), key=lambda kv: kv[1])
    spectrum_concentrated = bool(max_bin[1] >= 0.35)
    d2["concentration"] = {
        "worst16_max_bin": max_bin[0],
        "worst16_max_C_b": max_bin[1],
        "stable_concentrated": spectrum_concentrated,
    }
    write_json(out_dir / "d2_spectrum.json", d2)

    # D3 features
    feat_names = ["q_t", "enstrophy", "grad_energy", "hf_energy_ratio", "vmax", "vmin"]
    feat_mat = {k: np.array([r["features"][k] for r in rows_a], dtype=np.float64) for k in feat_names}
    corrs = {k: spearman_rho(feat_mat[k], e1) for k in feat_names}
    # quantile means of e1 by q_t
    q_order = np.argsort(feat_mat["q_t"])
    q_bins = {}
    for name, sl in [
        ("q_low", q_order[:32]),
        ("q_mid", q_order[48:80]),
        ("q_high", q_order[-32:]),
    ]:
        q_bins[name] = {
            "mean_e1": float(e1[sl].mean()),
            "mean_q_t": float(feat_mat["q_t"][sl].mean()),
        }
    d3 = {
        "spearman_vs_e1": corrs,
        "e1_by_q_t_quantile": q_bins,
        "top_feature": max(corrs.items(), key=lambda kv: abs(kv[1]) if not math.isnan(kv[1]) else -1)[0],
    }
    write_json(out_dir / "d3_features.json", d3)

    # D4 paired near-miss
    e1_r10 = np.array([r["e1"] for r in rows_b], dtype=np.float64)
    d_i = e1_r10 - e1  # negative => r10 better
    d_mean, d_lo, d_hi = paired_bootstrap_ci(d_i, seed=args.seed + 1)
    improved = d_i < 0
    improve_frac = float(improved.mean())
    # contribution of top-10 most improved (most negative d_i) to total mean improvement
    total_sum = float(d_i.sum())
    top10_idx = list(np.argsort(d_i)[:10])  # most negative
    top10_sum = float(d_i[top10_idx].sum())
    top10_contrib = abs(top10_sum / total_sum) if abs(total_sum) > 1e-15 else float("nan")
    # worst-decile of r9: do they worsen under r10?
    worst_decile = order_e1[-max(1, len(order_e1) // 10) :]
    worst_decile_mean_d = float(d_i[worst_decile].mean())
    worst_decile_worsened = bool(worst_decile_mean_d > 0)
    # few-sample dominance: if top-3 absolute contribution >= 50% of |total|
    top3_sum = float(d_i[list(np.argsort(d_i)[:3])].sum())
    few_sample = bool(abs(top3_sum) >= 0.5 * abs(total_sum)) if abs(total_sum) > 1e-15 else True

    # INCUBATE if CI entirely negative (credible improvement) but not promote-gate
    incubate = bool(d_hi < 0 and improve_frac >= 0.55 and not few_sample)
    near_miss_label = "INCUBATE_WEAK_SIGNAL" if incubate else "NOISE"

    d4 = {
        "definition": "d_i = e1_r10 - e1_r9  (negative means r10 better)",
        "mean_d": d_mean,
        "median_d": float(np.median(d_i)),
        "bootstrap_mean_ci95": [d_lo, d_hi],
        "improve_fraction": improve_frac,
        "top10_contribution_to_total_sum": top10_contrib,
        "top3_dominates_half_of_total": few_sample,
        "worst_decile_mean_d": worst_decile_mean_d,
        "worst_decile_worsened": worst_decile_worsened,
        "mean_e1_r9": float(e1.mean()),
        "mean_e1_r10": float(e1_r10.mean()),
        "label": near_miss_label,
        "note": "INCUBATE does not promote / does not enter report v-chain",
    }
    write_json(out_dir / "d4_paired_near_miss.json", d4)

    # Verdict
    stop_reasons = []
    if not ar_pass:
        stop_reasons.append("AR_preregister_failed")
    if half_conflict:
        stop_reasons.append("half_split_conflict")
    if not spectrum_concentrated:
        stop_reasons.append("spectrum_not_stably_concentrated")
    if few_sample:
        stop_reasons.append("near_miss_dominated_by_few_samples")

    if ar_pass and not half_conflict:
        verdict = "CONDITIONAL_PF_ALLOWED"
        headline = (
            "AR preregister conditions ALL passed; PF may be considered in a future Go — "
            "NOT auto-started this round."
        )
    else:
        verdict = "STOP_PRECISION_DEFENSE_ONLY"
        headline = (
            "AR exposure 可存在于 rollout，但不能解释正式单步尾部。"
            "永久停精度，只答辩。"
        )

    verdict_obj = {
        "verdict": verdict,
        "headline": headline,
        "ar_preregister_all_pass": ar_pass,
        "stop_reasons": stop_reasons,
        "near_miss_label": near_miss_label,
        "main_report_unchanged": {
            "relative_l2": 0.03530218452215195,
            "tag": "freeze_r9",
            "report": "v8",
        },
        "no_train_this_round": True,
        "generated_at": d0["generated_at"],
    }
    write_json(out_dir / "verdict.json", verdict_obj)

    # Figures
    # Write figures once under results/figures (+ demo/media); avoid duplicating PNGs in out_dir
    fig_dir = SUBMISSION / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_protocol(fig_dir / "protocol_vs_fno_paper_0128.png")
    plot_error_decomp(rows_a, fig_dir / "error_decomp_e1_tf_ar.png")
    heat = plot_spectrum_strip(rows_a, order_e1, fig_dir / "spectrum_best_median_worst.png")
    fig_names = [
        "protocol_vs_fno_paper_0128.png",
        "error_decomp_e1_tf_ar.png",
        "spectrum_best_median_worst.png",
        heat.name,
    ]
    mirror_figures(fig_dir, fig_names)

    # VERDICT.md
    md = f"""# Error Autopsy VERDICT · 2026-08-04

## 裁决

**`{verdict}`**

{headline}

| 检查项 | 结果 |
|--------|------|
| cond1 median(g)>0 且 CI_lo>0 | `{cond1}` (median_g={float(np.median(g)):.6g}, CI=[{g_lo:.6g},{g_hi:.6g}]) |
| cond2 ρ(e1,g)≥0.4 | `{cond2}` (ρ={rho:.4f}) |
| cond3 worst-16 重合≥8 | `{cond3}` (overlap={overlap}) |
| half-split conflict | `{half_conflict}` |
| spectrum concentrated | `{spectrum_concentrated}` (max {max_bin[0]} C_b={max_bin[1]:.3f}) |
| near-miss label | `{near_miss_label}` |

## 主报（未改）

- 公开 NS64 L2 **0.035302** · `freeze_r9` · 评测报告 **v8**
- 本轮 epochs=0；未 promote；未跑 `test_perf`

## 关键数字（freeze_r9）

| 指标 | 值 |
|------|-----|
| mean e1 | {float(e1.mean()):.6f} |
| mean e2_TF | {float(e2_tf.mean()):.6f} |
| mean e2_AR | {float(e2_ar.mean()):.6f} |
| mean / median g | {g_mean:.6f} / {float(np.median(g)):.6f} |
| top feature vs e1 | {d3['top_feature']} (ρ={corrs[d3['top_feature']]:.3f}) |
| r10−r9 mean d | {d_mean:.6g} · improve_frac={improve_frac:.3f} |

## 产物

- JSON: `d0_protocol.json` … `d4_paired_near_miss.json`, `per_sample_r9.json`, `verdict.json`
- 图: `protocol_vs_fno_paper_0128.png`, `error_decomp_e1_tf_ar.png`, `spectrum_best_median_worst.png`
- 镜像: `results/figures/` · `demo/media/`

## 纪律

- 即使 `CONDITIONAL_PF_ALLOWED`，本轮也**不开训**（需另开 Go）。
- `INCUBATE_WEAK_SIGNAL` **不进**评测报告 v 链。
"""
    (out_dir / "VERDICT.md").write_text(md)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt-a",
        type=str,
        default="checkpoints/fno_ns_public_freeze_r9_best.pt",
    )
    p.add_argument(
        "--ckpt-b",
        type=str,
        default="checkpoints/fno_ns_public_freeze_r10_best.pt",
    )
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260722)
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/run_logs/error_autopsy_20260804",
    )
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    if args.device != "cpu":
        print(f"[warn] plan mandates CPU; got --device={args.device}, still using CPU forwards", flush=True)
    out = run(args)
    print(f"DONE → {out}", flush=True)


if __name__ == "__main__":
    main()
