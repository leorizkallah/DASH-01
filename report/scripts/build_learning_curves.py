"""Build learning-curve figures and a budget audit from archived local metrics.

Run from the repository root: .venv/Scripts/python report/scripts/build_learning_curves.py
No cluster access, policy rollout, training, or presentation modification.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "report/figures"
OUT = ROOT / "report/data/learning-curves"
RUNS = [
    ("walk01", "WALK-01", "walk01_20261019_v1", "#5A626D"),
    ("flatfeet", "Flat-feet baseline", "dash01_flatfeet_symmetry_v1", "#E2001A"),
    ("clearance01", "CLEARANCE-01", "clearance01_20261025_v1", "#168BD2"),
]
SIDES = [("Left", "#168BD2"), ("Right", "#EF8018")]
SOURCES = {}


def source(path):
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8-sig")


def read_csv(path):
    return list(csv.DictReader(source(path).splitlines()))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, note):
    fig.text(0.01, 0.012, note, fontsize=9, color="#5A626D")
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180)
    plt.close(fig)


def smooth_blocks(x, y, block, weights=None):
    """Non-overlapping means; incomplete final block retained, no padding."""
    xs, ys = [], []
    for start in range(0, len(x), block):
        xx, yy = x[start : start + block], y[start : start + block]
        ww = np.ones(len(xx)) if weights is None else weights[start : start + block]
        valid = np.isfinite(yy) & np.isfinite(ww) & (ww > 0)
        xs.append(float(xx.mean()))
        ys.append(float(np.average(yy[valid], weights=ww[valid])) if valid.any() else np.nan)
    return xs, ys


def style(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold"})
    runs, milestones, evaluation_rows = [], [], []
    for key, label, folder, color in RUNS:
        path = ROOT / "docs/experiments" / folder
        progress = read_csv(path / "progress.csv")
        evaluations = read_csv(path / "eval.csv")
        config = json.loads(source(path / "resolved_config.json"))
        steps = column(progress, "time/env_steps")
        batch = config["n_envs"] * config["config"]["n_steps"]
        assert np.all(np.diff(steps) == batch) and steps[0] == batch
        assert steps[-1] >= config["total_steps"]
        assert np.all(np.diff(column(evaluations, "step")) > 0)
        assert all(0 <= float(r["alive_frac"]) <= 1 for r in evaluations)
        assert all(np.isclose(float(r["alive_frac"]), 1 - float(r["falls"]) / float(r["n"])) for r in evaluations)
        summary = dict(
            run=key, label=label, experiment=folder, actual_steps=int(steps[-1]),
            updates=len(progress), n_envs=config["n_envs"], rollout_ticks=config["config"]["n_steps"],
            transitions_per_update=batch, control_dt_s=0.01,
            pooled_simulated_hours=float(steps[-1] * 0.01 / 3600),
            first_evaluated_zero_physical_falls_step=next(
                (int(r["step"]) for r in evaluations if float(r["alive_frac"]) == 1), None),
        )
        best = None
        if (path / "best_eval.json").exists():
            best = json.loads(source(path / "best_eval.json"))
            matching = next(r for r in evaluations if int(r["step"]) == best["step"])
            assert np.isclose(float(matching["track_err"]), best["track_err_mean"])
            summary.update(selected_checkpoint_step=best["step"], selected_error_mps=best["track_err_mean"],
                           selected_mean_post_warmup_ticks=best["settled_ticks"])
            assert best["settled_ticks"] == 2500  # maximum 30 s minus 5 s, all selected trials
            for budget in (25, 50, 100, 150, 175, 200):
                eligible = [r for r in evaluations if int(r["step"]) <= budget * 1e6]
                no_fall = [r for r in eligible if float(r["alive_frac"]) == 1]
                lowest = min(no_fall, key=lambda r: float(r["track_err"])) if no_fall else None
                last = eligible[-1] if eligible else None
                milestones.append(dict(
                    run=key, budget_million=budget, latest_eval_step=int(last["step"]) if last else "",
                    latest_error_mps=float(last["track_err"]) if last else "",
                    latest_no_physical_fall_fraction=float(last["alive_frac"]) if last else "",
                    min_error_without_physical_falls_mps=float(lowest["track_err"]) if lowest else "",
                    min_error_without_physical_falls_step=int(lowest["step"]) if lowest else "",
                ))
        for r in evaluations:
            evaluation_rows.append(dict(run=key, step=int(r["step"]), error_mps=float(r["track_err"]),
                                        trials=int(r["n"]), physical_falls=int(r["falls"]),
                                        no_physical_fall_fraction=float(r["alive_frac"])))
        # Report the known diagnostic gap rather than silently filling it.
        diagnostic_gaps = {}
        for field in ("rollout/reward_mean", "clearance/left_mean_air_mm", "clearance/right_mean_air_mm",
                      "reward_terms/low_swing_clearance", "reward_terms/extra_swing_clearance"):
            if field in progress[0]:
                diagnostic_gaps[field] = [int(v) for v in steps[~np.isfinite(column(progress, field))]]
        summary["nonfinite_diagnostics_steps"] = diagnostic_gaps
        curriculum = {}
        for title, field, threshold, above in (
            ("shaping_at_least_0.99", "curriculum/shape_scale", 0.99, True),
            ("tracking_sigma_at_most_0.301", "curriculum/track_sigma", 0.301, False),
            ("efficiency_at_least_0.5", "curriculum/eff_scale", 0.5, True),
            ("command_upper_at_least_99pct", "curriculum/cmd_hi", 0.99, True),
        ):
            values = column(progress, field)
            indices = np.flatnonzero(values >= threshold if above else values <= threshold)
            curriculum[title] = int(steps[indices[0]]) if indices.size else None
        summary["observed_curriculum_milestones_steps"] = curriculum
        runs.append(dict(key=key, label=label, color=color, path=path, progress=progress,
                         evaluations=evaluations, config=config, best=best, summary=summary))

    clearance = [json.loads(s) for s in source(runs[-1]["path"] / "clearance_evaluations.jsonl").splitlines()]
    assert clearance[0]["step"] == 0 and clearance[-1]["step"] == runs[-1]["summary"]["actual_steps"]
    assert np.all(np.diff([r["step"] for r in clearance]) > 0)
    assert all(r["trials"] == 12 and r["target_mps"] == 1.5 for r in clearance)
    clearance_rows = [dict(
        step=r["step"], mean_air_left_mm=r["mean_air_clearance_mm"][0],
        mean_air_right_mm=r["mean_air_clearance_mm"][1],
        midswing_ge20_left_fraction=r["midswing_ge20mm_fraction"][0],
        midswing_ge20_right_fraction=r["midswing_ge20mm_fraction"][1],
        full_duration_survival_fraction=r["survival_fraction"], qualified_fraction=r["qualified_fraction"],
        error_mps=r["track_error_mps"],
    ) for r in clearance]
    write_csv("periodic_evaluations.csv", evaluation_rows)
    write_csv("budget_milestones.csv", milestones)
    write_csv("clearance_evaluations.csv", clearance_rows)

    # A 16:9 slide overview: task outcomes, rather than incomparable reward scales.
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.1))
    fig.suptitle("What did the training budgets achieve?", fontsize=20, fontweight="bold")
    for run in runs[:2]:
        rows, best = run["evaluations"], run["best"]
        x = column(rows, "step") / 1e6
        axes[0, 0].plot(x, column(rows, "track_err"), "o-", ms=4, color=run["color"], label=run["label"])
        axes[0, 0].scatter(best["step"] / 1e6, best["track_err_mean"], s=150, marker="*", color=run["color"], zorder=5)
        axes[0, 1].plot(x, 100 * column(rows, "alive_frac"), "o-", ms=4, color=run["color"], label=run["label"])
    axes[0, 0].set(title="Fresh policies: nominal speed tracking", ylabel="Mean absolute error (m/s)", ylim=(0, 1.32))
    axes[0, 1].set(title="Fresh policies: trials without physical falls", ylabel="Trials (%)", ylim=(-3, 105))
    for ax in axes[0]:
        style(ax, "Training transitions (millions)")
        ax.set_xlim(0, 200)
        ax.legend(fontsize=10)
    x = np.array([r["step"] for r in clearance]) / 1e6
    for side, (name, color) in enumerate(SIDES):
        axes[1, 0].plot(x, [r["mean_air_clearance_mm"][side] for r in clearance], "o-", ms=4, color=color, label=name)
        axes[1, 1].plot(x, [100 * r["midswing_ge20mm_fraction"][side] for r in clearance], "o-", ms=4, color=color, label=name)
    axes[1, 0].axhline(20, color="#5A626D", ls="--", label="20 mm evaluation threshold")
    axes[1, 1].axhline(80, color="#5A626D", ls="--", label="80% evaluation criterion")
    axes[1, 0].set(title="50M fine-tuning: mean airborne clearance", ylabel="Whole-foot height (mm)", ylim=(0, 27))
    axes[1, 1].set(title="50M fine-tuning: mid-swing samples at least 20 mm", ylabel="Scheduled mid-swing samples (%)", ylim=(-3, 105))
    for ax in axes[1]:
        style(ax, "Additional transitions from flat-feet best (millions)")
        ax.set_xlim(0, 50.1)
        ax.legend(fontsize=9)
    save(fig, "learning_curves_overview",
         "Top: 28 trials / 7 commands; stars = saved keepers. No fall does not exclude early symmetry termination.\n"
         "Bottom: 12 trials at 1.50 m/s. Raw evaluation points, one training seed. No claim of full convergence.")

    # Per-objective detail: never compare scalar returns from different rewards.
    for run in runs:
        p, best = run["progress"], run["best"]
        x = column(p, "time/env_steps") / 1e6
        block = 50 if run["key"] == "clearance01" else 100
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3))
        fig.suptitle(f'{run["label"]}: learning and evaluation', fontsize=19, fontweight="bold")
        for ax, field, scale, title, ylabel, weights in (
            (axes[0, 0], "rollout/reward_mean", 1, "Training reward per control transition", "Reward (this objective only)", None),
            (axes[0, 1], "rollout/ep_len_mean", 0.01, "Mean duration of completed training episodes", "Seconds", column(p, "rollout/episodes")),
        ):
            y = column(p, field) * scale
            raw = y.copy()
            if weights is not None:
                raw[weights == 0] = np.nan  # logger otherwise falls back to unfinished episode lengths
            ax.plot(x, raw, color=run["color"], alpha=0.12, lw=0.4, rasterized=True)
            xx, yy = smooth_blocks(x, raw, block, weights)
            ax.plot(xx, yy, color=run["color"], lw=2, label=f"{block}-update block mean")
            ax.set(title=title, ylabel=ylabel)
            ax.legend(fontsize=9)
        axes[0, 1].set_ylim(0, 31)
        if best:
            ev = run["evaluations"]
            ex = column(ev, "step") / 1e6
            ey = column(ev, "track_err")
            axes[1, 0].plot(ex, ey, "o-", color=run["color"], ms=4)
            axes[1, 0].scatter(best["step"] / 1e6, best["track_err_mean"], marker="*", s=180, color="#26323F", label="Saved keeper", zorder=5)
            axes[1, 1].plot(ex, 100 * column(ev, "alive_frac"), "o-", color=run["color"], ms=4)
            axes[1, 1].set(title="Nominal evaluation: without physical falls", ylabel="Trials (%)", ylim=(-3, 105))
            for ax in axes.flat:
                ax.axvline(best["step"] / 1e6, color="#26323F", ls=":", alpha=0.6)
            axes[1, 0].legend(fontsize=9)
        else:
            ex = np.array([r["step"] for r in clearance]) / 1e6
            axes[1, 0].plot(ex, [r["track_error_mps"] for r in clearance], "o-", color=run["color"])
            axes[1, 1].plot(ex, [100 * r["survival_fraction"] for r in clearance], "o-", label="Full 30 s duration", color=run["color"])
            axes[1, 1].plot(ex, [100 * r["qualified_fraction"] for r in clearance], "s--", label="All clearance criteria", color="#E2001A")
            axes[1, 1].set(title="Nominal evaluation: survival and qualification", ylabel="Trials (%)", ylim=(-3, 105))
            axes[1, 1].legend(fontsize=9)
        axes[1, 0].set(title="Nominal evaluation: speed tracking", ylabel="Mean absolute error (m/s)")
        xlabel = "Additional training transitions (millions)" if not best else "Training transitions (millions)"
        for ax in axes.flat:
            style(ax, xlabel)
            ax.set_xlim(0, max(x))
        save(fig, f'learning_curves_{run["key"]}',
             "Faint: raw rollout values; episode means weighted by completed-episode counts. Reward changes with curriculum.\n"
             + ("28 nominal trials, commands 0.25-2.50 m/s. No physical fall is not proof of full duration or clean gait."
                if best else "Warm start from flat-feet best at 184.32M, fresh step counter. 12 nominal trials at 1.50 m/s."))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3))
    fig.suptitle("Difficulty continued changing late in training", fontsize=19, fontweight="bold")
    for run in runs[:2]:
        p = run["progress"]
        x = column(p, "time/env_steps") / 1e6
        for ax, field, multiplier, title, ylabel in (
            (axes[0, 0], "curriculum/cmd_hi", run["config"]["config"]["v_max"], "Upper bound of training speed commands", "m/s"),
            (axes[0, 1], "curriculum/shape_scale", 1, "Reward shaping scale", "Scale"),
            (axes[1, 0], "curriculum/eff_scale", 1, "Efficiency curriculum scale", "Scale"),
            (axes[1, 1], "curriculum/dr_scale", 1, "Domain randomization scale", "Scale"),
        ):
            ax.plot(x, column(p, field) * multiplier, color=run["color"], label=run["label"])
            ax.set(title=title, ylabel=ylabel, xlim=(0, 200))
        axes[0, 0].plot(x, column(p, "curriculum/cmd_lo") * run["config"]["config"]["v_max"], color=run["color"], ls=":", alpha=0.65)
    for ax in axes.flat:
        style(ax, "Training transitions (millions)")
        ax.legend(fontsize=10)
    save(fig, "learning_curves_curriculum",
         "Logged curriculum values, not nominal configured durations. Dotted command curves = lower bounds.\n"
         "Gating can delay progression; changing difficulty prevents treating reward alone as a stationary convergence test.")

    summary = dict(runs=[r["summary"] for r in runs], clearance_start=clearance_rows[0],
                   clearance_final=clearance_rows[-1],
                   periodic_alive_fraction_definition="1 - physical_falls / n; may include early non-fall endings",
                   smoothing="Non-overlapping 100-update means (fresh) / 50-update means (fine-tune); episode-count weighting for durations",
                   caveats=["One training seed per run", "No independent test-seed or repeated-seed convergence study",
                            "Rewards differ and curricula change", "Periodic evaluation is sparse and can miss better checkpoints",
                            "Nominal simulation does not validate hardware or walking quality"], sources_sha256=SOURCES)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for rel, digest in SOURCES.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
    write_report(runs, clearance_rows, milestones)
    print(json.dumps({"figures": 5, "formats": ["png", "svg", "pdf"], "input_files_verified": len(SOURCES),
                      "actual_steps": {r["key"]: r["summary"]["actual_steps"] for r in runs}}, indent=2))


def write_report(runs, clearance, milestones):
    lines = [
        "# Learning curves and training budgets", "",
        "Built from archived local metrics on 27 October 2026. No training or cluster monitoring was performed.", "",
        "The curves support allowing long training budgets for exploration and curriculum progression. "
        "They do **not** establish that 200M steps is optimal, or that these policies have fully converged.", "",
        "![Learning curves overview](figures/learning_curves_overview.png)", "",
        "[Slide-ready PNG](figures/learning_curves_overview.png) · [Vector PDF](figures/learning_curves_overview.pdf) · "
        "[Editable SVG](figures/learning_curves_overview.svg)", "",
        "## Evidence for the budgets", "",
        "| Run | Total transitions | First evaluated checkpoint with zero physical falls | Selected saved keeper | Keeper error |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs[:2]:
        s = r["summary"]
        lines.append(f'| {r["label"]} | {s["actual_steps"]/1e6:.2f}M | '
                     f'{s["first_evaluated_zero_physical_falls_step"]/1e6:.2f}M | '
                     f'{s["selected_checkpoint_step"]/1e6:.2f}M | {s["selected_error_mps"]:.4f} m/s |')
    lines += ["", "**Metric correction:** the archived evaluator defines `alive_frac = 1 - physical_falls / n`. "
              "A symmetry termination can end a trial early without being counted as a physical fall. "
              "Therefore the periodic curves show **trials without physical falls**, not certified full-duration survival. "
              "Both selected keepers have a mean of 2,500 evaluated post-warmup ticks, the maximum after the 5 s "
              "warmup in a 30 s trial; this supports full duration for all 28 trials at those two checkpoints. "
              "It does not certify clean, symmetric walking.", "",
              "Neither fresh run had an evaluation with zero physical falls by 100M transitions. "
              "For the flat-feet run, the lowest evaluated speed error among checkpoints without physical falls "
              "was 0.0588 m/s by 150M and 0.0489 m/s at 184.32M (about 17% lower). "
              "The last periodic checkpoint at 199.07M had a lower error, 0.0442 m/s, but one physical fall. "
              "More training is not monotonically better; retaining a validated keeper matters.", "",
              "| Run | Budget | Last evaluation within budget | Speed error | Trials without physical falls |",
              "|---|---:|---:|---:|---:|"]
    for m in milestones:
        if m["budget_million"] in (50, 100, 150, 200):
            lines.append(f'| {m["run"]} | {m["budget_million"]}M | {m["latest_eval_step"]/1e6:.2f}M | '
                         f'{m["latest_error_mps"]:.4f} m/s | {m["latest_no_physical_fall_fraction"]*100:.1f}% |')
    lines += ["", "These are the last evaluations within each budget, **not** a cumulative best curve. "
              "All raw evaluation points remain visible in the figures. The CSV also includes the lowest-error "
              "checkpoint without physical falls within each budget; this diagnostic is not the actual keeper rule "
              "(which also uses heading and other criteria).", "",
              "## Why the curriculum needs time", "",
              "![Curriculum progression](figures/learning_curves_curriculum.png)", "",
              "Both fresh runs reached a shaping scale of at least 0.99 at 84.75M and an efficiency scale "
              "of at least 0.5 at 120.05M. The upper command bound reached 99% of its 2.5 m/s maximum "
              "only at 189.63M in WALK-01 and 182.22M in flat-feet. "
              "These are observed first threshold crossings, not proof that a curriculum completed or stayed there. "
              "Gated curricula can advance much later than their nominal configured step durations. "
              "A short run would train on a different difficulty mix.", "",
              "## The 50M clearance experiment was still changing", ""]
    b, f = clearance[0], clearance[-1]
    lines += [f'At 1.50 m/s, mean whole-foot clearance while airborne increased from '
              f'**{b["mean_air_left_mm"]:.2f}/{b["mean_air_right_mm"]:.2f} mm** to '
              f'**{f["mean_air_left_mm"]:.2f}/{f["mean_air_right_mm"]:.2f} mm** (left/right). '
              'The curves continue changing in the last several million transitions. '
              f'Final scheduled mid-swing coverage above 20 mm is '
              f'**{f["midswing_ge20_left_fraction"]*100:.1f}%/{f["midswing_ge20_right_fraction"]*100:.1f}%**, '
              'below the 80% criterion for each foot. All 12 nominal trials reach 30 s, but none satisfies all '
              'clearance criteria. Good mean height alone does not establish consistent swing clearance.', "",
              "This is **50M additional transitions** from flat-feet's selected 184.32M checkpoint. "
              "The plotted counter restarts at zero; it is not fresh training. "
              "It supports testing a larger fine-tuning budget, while the new CLEARANCE-02 reward uses different "
              "10–25 mm thresholds. Its 200M budget is an experimental allowance, not an already demonstrated requirement. "
              "The first CLEARANCE-02 attempt performed no main training, so it has no learning curve to include.", "",
              "## Detailed learning curves", ""]
    for r in runs:
        lines += [f'### {r["label"]}', "", f'![{r["label"]}](figures/learning_curves_{r["key"]}.png)', "",
                  f'[PNG](figures/learning_curves_{r["key"]}.png) · [PDF](figures/learning_curves_{r["key"]}.pdf) · '
                  f'[SVG](figures/learning_curves_{r["key"]}.svg)', ""]
    lines += ["## What a step means", "",
              "One step is one control transition for one simulated robot at **100 Hz** (10 ms). "
              "There are **512 parallel environments**, each collecting 36 control ticks per rollout: "
              "**18,432 transitions per PPO update**. The 200M requests round up to **200,005,632 transitions / "
              "10,851 updates**; 50M rounds up to **50,006,016 / 2,713 updates**. "
              "These are neither episode counts nor optimizer minibatch counts. PPO reuses each rollout over multiple epochs.", "",
              "200M transitions correspond to roughly **556 pooled simulated hours**, or about **65 simulated minutes "
              "per environment**, spread across many resets. The physics advances in 1 ms substeps. "
              "Simulation time, GPU wall time and the number of independent training seeds are different quantities.", "",
              "## Plotting and evaluation protocol", "",
              "- Fresh runs: 28 nominal first episodes, four trials at each command 0.25, 0.50, 0.75, 1.00, "
              "1.50, 2.00 and 2.50 m/s; 30 s maximum, first 5 s excluded from speed error. "
              "Evaluations every 400 rollouts (7.3728M transitions). No initial zero-step evaluation is available.",
              "- CLEARANCE-01: 12 nominal first episodes at 1.50 m/s, including a matched zero-step baseline "
              "and final evaluation. The custom JSONL records full-duration survival separately from physical falls.",
              "- Evaluation points are unsmoothed. Lines only connect sampled checkpoints. The same nominal "
              "evaluation protocol/seed is reused for selection; this is not an independent held-out robustness study.",
              "- Faint training curves are raw rollout logs. Bold curves use non-overlapping block means: "
              "100 updates (1.8432M transitions) for fresh runs, 50 updates (0.9216M) for fine-tuning. "
              "Final partial blocks are retained. Completed-episode duration means are weighted by episode count; "
              "rollouts with no completed episodes are excluded from that duration average.",
              "- Reward is the average per control transition, not cumulative episode return. "
              "Reward definitions differ between runs and curricula change within runs, so scalar reward "
              "levels cannot rank the policies or by themselves prove convergence.",
              "- At 14,321,664 CLEARANCE-01 transitions, several clearance diagnostics are non-finite while "
              "the rollout mean reward is finite. The source logs are preserved; no interpolation or data repair is performed. "
              "The clearance curves here use separate finite evaluation records.",
              "- One training seed per run. No confidence intervals, optimal-budget claim or convergence-rate "
              "significance claim is supported. Nominal survival/tracking does not rule out sliding, contact chatter, "
              "gait asymmetry, startup failures in other replay setups, or a sim-to-real gap.", "",
              "## Suggested presentation wording", "",
              "> We allowed 200 million pooled control transitions because the fresh policies and gated curricula "
              "were still developing beyond 100 million. The flat-feet baseline's selected checkpoint was reached "
              "at 184 million transitions. The 50-million-transition clearance fine-tuning showed that foot lift "
              "could improve, but it was still changing and had not met all clearance criteria. These curves "
              "justify the experimental budgets; they do not prove full convergence or that more training is always better.", "",
              "## Reproduce and trace the data", "",
              "```powershell", ".venv/Scripts/python report/scripts/build_learning_curves.py", "```", "",
              "[Summary and input SHA-256 hashes](data/learning-curves/summary.json) · "
              "[Budget milestones CSV](data/learning-curves/budget_milestones.csv) · "
              "[Periodic evaluations CSV](data/learning-curves/periodic_evaluations.csv) · "
              "[Clearance evaluations CSV](data/learning-curves/clearance_evaluations.csv)", "",
              "Original per-run `progress.csv`, `eval.csv`, `resolved_config.json`, selected `best_eval.json` "
              "and `clearance_evaluations.jsonl` remain under `docs/experiments/`. "
              "Input hashes are verified unchanged after generation. Existing presentations are untouched.", ""]
    (ROOT / "report/learning-curves.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
