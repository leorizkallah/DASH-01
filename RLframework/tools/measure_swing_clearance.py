"""Measure whole-foot swing clearance from saved MJX poses without re-simulation.

The recorder saves qpos before a control transition and contact flags after it.
Align pose k with contact k-1, discard startup and never include auto-resets.
"""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/experiments/dash01_flatfeet_symmetry_v1"
OUT = ROOT / "report/data/flatfeet-clearance"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pad_clearances(model, qpos):
    """Signed distance of the lowest pad corner to the actual floor plane [T,L/R,toe/heel]."""
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    pads = np.array(
        [
            [model.geom(f"{pad}_{side}_col").id for pad in ("foot", "heel")]
            for side in "LR"
        ]
    )
    assert np.all(model.geom_type[pads] == mujoco.mjtGeom.mjGEOM_BOX)
    values = []
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
        position = data.geom_xpos[pads]
        rotation = data.geom_xmat[pads].reshape(2, 2, 3, 3)
        support = np.sum(
            np.abs(np.einsum("i,abij->abj", normal, rotation)) * model.geom_size[pads],
            axis=-1,
        )
        height = (position - data.geom_xpos[floor]) @ normal - support
        if i in (0, len(qpos) // 2, len(qpos) - 1):
            # Independent geometric check: explicitly transform all eight box corners.
            for foot, pad in itertools.product(range(2), repeat=2):
                gid = pads[foot, pad]
                xyz = (corners * model.geom_size[gid]) @ rotation[
                    foot, pad
                ].T + position[foot, pad]
                explicit = np.min((xyz - data.geom_xpos[floor]) @ normal)
                np.testing.assert_allclose(height[foot, pad], explicit, atol=1e-10)
        values.append(height)
    return np.asarray(values)


def air_bouts(airborne):
    """Return sample intervals [start,end) and whether neither boundary was truncated."""
    edges = np.diff(np.r_[False, airborne, False].astype(int))
    return [
        (int(a), int(b), bool(a > 0 and b < len(airborne)))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def quantiles(values):
    return np.percentile(values, [10, 50, 90]).tolist() if len(values) else [None] * 3


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((DATA / "resolved_config.json").read_text())["config"]
    model_path = ROOT / "RLframework" / cfg["model_path"]
    manifest = json.loads((DATA / "source-manifest.json").read_text())
    assert sha(model_path) == manifest["files"][model_path.relative_to(ROOT).as_posix()]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    results, cycles, traces = [], [], {}
    for label in ("best", "final"):
        path = DATA / f"flatfeet_trace_{label}.npz"
        with np.load(path) as raw:
            meta = json.loads(str(raw["meta"].item()))
            for i, trial in enumerate(meta["trials"]):
                n, dt = trial["ticks"], meta["dt"]
                assert n <= len(raw["qpos"])
                # No dynamics/contact solver: only rigid-body forward kinematics.
                pads = pad_clearances(model, raw["qpos"][:n, i])[1:]
                t = np.arange(1, n) * dt
                grounded = raw["flatfeet_grounded"][: n - 1, i]
                np.testing.assert_array_equal(
                    grounded,
                    raw["flatfeet_toe"][: n - 1, i] | raw["flatfeet_heel"][: n - 1, i],
                )
                signed = pads.min(axis=-1)
                clearance = np.maximum(signed, 0.0)
                key = f"{label}_{trial['target']:.2f}"
                traces[key + "_time"] = t
                traces[key + "_pad_clearance"] = pads
                traces[key + "_grounded"] = grounded
                for foot, side in enumerate("LR"):
                    airborne = ~grounded[:, foot]
                    measured = (t >= 5.0) & airborne
                    samples = clearance[measured, foot] * 1000
                    assert samples.size
                    full_cycles = []
                    for start, end, complete in air_bouts(airborne):
                        if t[start] < 5.0:
                            continue
                        record = dict(
                            checkpoint=label,
                            target_mps=trial["target"],
                            foot=side,
                            start_s=float(t[start]),
                            end_s=float(t[end - 1] + dt),
                            duration_s=(end - start) * dt,
                            complete=complete,
                            peak_clearance_mm=float(
                                clearance[start:end, foot].max() * 1000
                            ),
                            min_signed_clearance_mm=float(
                                signed[start:end, foot].min() * 1000
                            ),
                            toe_peak_mm=float(
                                np.maximum(pads[start:end, foot, 0], 0).max() * 1000
                            ),
                            heel_peak_mm=float(
                                np.maximum(pads[start:end, foot, 1], 0).max() * 1000
                            ),
                        )
                        cycles.append(record)
                        if complete:
                            full_cycles.append(record)
                    substantial = [
                        c
                        for c in full_cycles
                        if c["duration_s"] + 1e-9 >= cfg["symmetry_min_phase_s"]
                    ]
                    peaks = [c["peak_clearance_mm"] for c in substantial]
                    p10, p50, p90 = quantiles(peaks)
                    result = dict(
                        checkpoint=label,
                        checkpoint_step=meta["step"],
                        target_mps=trial["target"],
                        achieved_mps=trial["mean_speed_after_5s"],
                        foot=side,
                        outcome=trial["outcome"],
                        analysis_end_s=trial["seconds"],
                        airborne_s=float(measured.sum() * dt),
                        mean_air_clearance_mm=float(samples.mean()),
                        median_air_clearance_mm=float(np.median(samples)),
                        max_air_clearance_mm=float(samples.max()),
                        signed_air_min_mm=float(signed[measured, foot].min() * 1000),
                        complete_swing_bouts=len(full_cycles),
                        complete_swing_bouts_ge40ms=len(substantial),
                        short_bout_fraction=float(
                            np.mean(
                                [c["duration_s"] < 0.04 - 1e-9 for c in full_cycles]
                            )
                        )
                        if full_cycles
                        else None,
                        peak_p10_mm=p10,
                        peak_median_mm=p50,
                        peak_p90_mm=p90,
                        mean_peak_mm=float(np.mean(peaks)) if peaks else None,
                        median_duration_ge40ms_s=float(
                            np.median([c["duration_s"] for c in substantial])
                        )
                        if substantial
                        else None,
                        airborne_time_below_5mm=float(np.mean(samples < 5.0)),
                        airborne_time_above_10mm=float(np.mean(samples >= 10.0)),
                        airborne_time_above_20mm=float(np.mean(samples >= 20.0)),
                    )
                    for threshold in (5, 10, 15, 20):
                        result[f"swing_peaks_ge{threshold}mm_fraction"] = (
                            float(np.mean(np.asarray(peaks) >= threshold))
                            if peaks
                            else None
                        )
                    results.append(result)
    write_csv(OUT / "summary.csv", results)
    focus = [r for r in results if r["target_mps"] == 1.5]
    write_csv(OUT / "summary-1.50mps.csv", focus)
    write_csv(OUT / "swing-bouts.csv", cycles)
    np.savez_compressed(OUT / "clearance-traces.npz", **traces)
    record = dict(
        date="2026-10-25",
        unit="mm unless explicitly seconds/metres",
        startup_excluded_s=5.0,
        focus_target_mps=1.5,
        focus_pooled_mean_air_clearance_mm={
            label: float(
                np.average(
                    [
                        r["mean_air_clearance_mm"]
                        for r in focus
                        if r["checkpoint"] == label
                    ],
                    weights=[
                        r["airborne_s"] for r in focus if r["checkpoint"] == label
                    ],
                )
            )
            for label in ("best", "final")
        },
        method="Minimum signed distance from either oriented toe/heel box corner to the floor. "
        "Summaries clip penetration to zero but also retain signed minima. "
        "Swing is neither force-qualified pad in contact (training threshold 2 N). "
        "Pose k is aligned with contact k-1. No physics resimulation. "
        "Peak statistics use complete swing bouts >=40 ms, independent of preceding stance validity; "
        "all-swing sample statistics retain short bouts. Partial boundary bouts excluded from peaks.",
        limitations="100 Hz sampling can miss sub-tick peaks/contact changes. One nominal trial per speed "
        "and checkpoint, not a robustness study. Below-threshold loading is not proof of a geometric gap.",
        rows=results,
        source_hashes={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                model_path,
                DATA / "flatfeet_trace_best.npz",
                DATA / "flatfeet_trace_final.npz",
                DATA / "resolved_config.json",
                Path(__file__),
            ]
        },
    )
    (OUT / "summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    plots(results, traces)
    print(json.dumps(results, indent=2))


def plots(rows, traces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#168bd2", "#ef8018"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(
        axs,
        ("mean_air_clearance_mm", "peak_median_mm"),
        (
            "Time-averaged clearance during swing",
            "Median maximum within a swing >=40 ms",
        ),
    ):
        for f, side in enumerate("LR"):
            values = [
                next(
                    r[key]
                    for r in rows
                    if r["checkpoint"] == label
                    and r["target_mps"] == 1.5
                    and r["foot"] == side
                )
                for label in ("best", "final")
            ]
            bars = ax.bar(
                np.arange(2) + (f - 0.5) * 0.34,
                values,
                width=0.32,
                color=colors[f],
                label="Left" if f == 0 else "Right",
            )
            ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Best, 184.32M", "Final, 200.01M"],
            ylabel="Whole-foot clearance (mm)",
            title=title,
        )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Focus: target 1.50 m/s; first 5 s excluded; each foot analysed only while airborne"
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"report/figures/flatfeet_clearance_1.50mps.{ext}", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for j, label in enumerate(("best", "final")):
        for i, speed in enumerate((0.25, 0.75, 1.5)):
            ax = axs[j, i]
            for f, side in enumerate("LR"):
                r = next(
                    r
                    for r in rows
                    if r["checkpoint"] == label
                    and r["target_mps"] == speed
                    and r["foot"] == side
                )
                mid = r["peak_median_mm"]
                ax.errorbar(
                    f,
                    mid,
                    yerr=[[mid - r["peak_p10_mm"]], [r["peak_p90_mm"] - mid]],
                    fmt="o",
                    capsize=7,
                    color=colors[f],
                    label=f"{side}: n={r['complete_swing_bouts_ge40ms']}",
                )
            ax.axhline(10, color="gray", ls=":", lw=0.8)
            ax.axhline(20, color="gray", ls="--", lw=0.8)
            ax.set(
                title=f"{label}: target {speed:.2f} m/s",
                xticks=[0, 1],
                xticklabels=["Left", "Right"],
                xlim=(-0.6, 1.6),
            )
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
    axs[0, 0].set_ylabel("Whole-foot peak clearance (mm)")
    axs[1, 0].set_ylabel("Whole-foot peak clearance (mm)")
    fig.suptitle(
        "Median swing peak and 10th–90th percentile range; complete air bouts >=40 ms, after 5 s"
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"report/figures/flatfeet_swing_clearance.{ext}", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for j, label in enumerate(("best", "final")):
        for i, speed in enumerate((0.25, 0.75, 1.5)):
            key = f"{label}_{speed:.2f}"
            t = traces[key + "_time"]
            pads = traces[key + "_pad_clearance"]
            ax = axs[j, i]
            win = (t >= 10) & (t <= 12)
            for f, side in enumerate("LR"):
                height = np.maximum(pads[:, f].min(axis=-1), 0) * 1000
                air = ~traces[key + "_grounded"][:, f]
                ax.plot(t[win], height[win], color=colors[f], alpha=0.35)
                ax.plot(
                    t[win],
                    np.where(air[win], height[win], np.nan),
                    color=colors[f],
                    lw=1.5,
                    label=side + " airborne",
                )
            ax.axhline(10, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{label}: {speed:.2f} m/s")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Lowest foot point above floor (mm)")
    fig.suptitle(
        "Recorded whole-foot clearance: strong lines indicate neither pad in force-qualified contact"
    )
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(
            ROOT / f"report/figures/flatfeet_swing_clearance_trace.{ext}", dpi=160
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
