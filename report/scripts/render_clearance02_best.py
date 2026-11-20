"""Render preselected trial 0 of the retained CLEARANCE-02 12-trial MJX replay."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/experiments/clearance02_20261027_v2"
sys.path.insert(0, str(ROOT / "RLframework/tools"))
from gait_video import render


def main():
    original = DATA / "clearance02_trace_best_eval12.npz"
    with np.load(original, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
        assert meta["step"] == 132710400
        assert meta["batch_size"] == 12 and len(meta["trials"]) == 12
        assert meta["command_protocol"] == "saved reset command"
        assert meta["checkpoint"] == "best_clearance_candidate.msgpack"
        # Trial 0 was chosen before recording, not selected by outcome or appearance.
        selected = dict(meta, trials=[meta["trials"][0]], selected_original_trial=0)
        selected["note"] = "Trial 0 of full retained batch 12. Rendering selection does not change simulation or checkpoint."
        arrays = {k: z[k][:, :1].copy() for k in z.files if k != "meta"}
    manifest = json.loads((DATA / "source-manifest.json").read_text())
    model_path = ROOT / "RLframework" / meta["model"]
    model_key = model_path.relative_to(ROOT).as_posix()
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest["files"][model_key]
    subset = DATA / "clearance02_trace_best_trial0.npz"
    np.savez_compressed(subset, meta=json.dumps(selected), **arrays)
    render(SimpleNamespace(trace=subset, out=ROOT / "report/videos", fast_render=True,
                           two_views=True, fps=20, prefix="clearance02_best",
                           label="CLEARANCE-02 candidate 132.71M | trial 0/12",
                           backend_label="MJX replay"))
    video = ROOT / "report/videos/clearance02_best_1.50mps.mp4"
    reader = imageio.get_reader(str(video))
    try:
        info = reader.get_meta_data()
        middle = reader.get_data(round(min(10, info["duration"] / 2) * info["fps"]))
        imageio.imwrite(DATA / "video-preview.jpg", middle)
    finally:
        reader.close()
    assert info["codec"] == "h264" and info["size"] == (1280, 720)
    assert abs(info["duration"] - meta["trials"][0]["seconds"]) <= 1 / 20 + .01
    provenance = dict(training_job=3232851, replay_job=3238862, checkpoint_step=meta["step"],
                      checkpoint_sha256=meta["checkpoint_sha256"], date="2026-11-02",
                      selected_trial=0, original_batch_size=12, command_protocol=meta["command_protocol"],
                      replay_trial=meta["trials"][0],
                      video=dict(duration_s=info["duration"], fps=info["fps"], size=info["size"], codec=info["codec"]),
                      files_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [original, original.with_suffix(".json"), subset, model_path, video,
                                              ROOT / "RLframework/tools/gait_video.py", Path(__file__)]},
                      limitations="Nominal replay, not hardware validation. One preselected trial shown; all 12 trial metadata retained."
                      " Independent standalone startup discrepancy not tested by this recording.")
    (DATA / "replay-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["video"], indent=2))


if __name__ == "__main__":
    main()
