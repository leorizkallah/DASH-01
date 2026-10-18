"""Render the CPU standing validation and plot its measured trace (not a policy)."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dash_report_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import imageio.v2 as imageio

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parent


def main():
    model = mujoco.MjModel.from_xml_path(str(REPO / "RLframework/model/dash01_cad_v2_free.xml"))
    trace = np.load(REPO / "dash-01CAD-v2/generated/standing_trace.npz")["trace"]
    q = trace[:, 1:1+model.nq]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(trace[:, 0], (q[:, 2]-q[0, 2])*1000)
    axes[0].set_ylabel("Height change (mm)")
    axes[1].plot(trace[:, 0], np.degrees(q[:, 3]), label="Roll")
    axes[1].plot(trace[:, 0], np.degrees(q[:, 4]), label="Pitch")
    axes[1].set_ylabel("Body angle (deg)")
    axes[1].legend()
    for i, name in enumerate(("Hip L", "Cam L", "Thigh L", "Hip R", "Cam R", "Thigh R")):
        axes[2].plot(trace[:, 0], trace[:, -6+i], label=name)
    axes[2].set_ylabel("Motor torque (Nm)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("CAD v2: five-second standing gate\nConstant motor targets, free base; no learned gait")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(REPORT / f"figures/gait01_standing_gate.{ext}", dpi=160)
    plt.close(fig)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.02, 0, .43]
    camera.distance, camera.azimuth, camera.elevation = 1.65, 130, -12
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(REPORT / "videos/cad_v2_standing_gate.mp4", fps=30, codec="libx264", quality=8) as writer:
            for t in np.arange(0, 5, 1/30):
                i = min(int(np.searchsorted(trace[:, 0], t)), len(q)-1)
                data.qpos[:] = q[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    print("Saved standing trace plots and five-second CPU validation video.")


if __name__ == "__main__":
    main()
