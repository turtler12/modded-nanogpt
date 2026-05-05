"""
Plot val loss curves for muon_speedrun experiments vs SOTA baselines.

Baselines (from records/track_3_optimization):
  - Muon (#12, 3325 steps)
  - NorMuon (#10, 3250 steps)
  - ContraNorMuon (#11 SOTA, 3225 steps)

My runs (synced from RunPod):
  - Any .txt files in runs/ directory

Usage:
  python plot.py
"""
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LOG_PATTERN = re.compile(r'^step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+train_time:([0-9.]+)s')

def parse_log(path):
    steps, losses, times = [], [], []
    for line in Path(path).read_text().splitlines():
        m = LOG_PATTERN.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            times.append(float(m.group(3)))
    return steps, losses, times

def smooth(losses, k=1):
    """Simple moving average for readability."""
    if k <= 1:
        return losses
    out = []
    for i, v in enumerate(losses):
        window = losses[max(0, i-k+1):i+1]
        out.append(sum(window) / len(window))
    return out

HERE = Path(__file__).parent

baselines = {
    "Muon (3325 steps, #12)":         HERE / "sota_baseline/muon_baseline.txt",
    "NorMuon (3250 steps, #10)":       HERE / "sota_baseline/normuon_baseline.txt",
    "ContraNorMuon (3225 steps, SOTA)": HERE / "sota_baseline/contra_muon_sota.txt",
}

my_runs = sorted((HERE / "runs").glob("**/*.txt")) if (HERE / "runs").exists() else []

baseline_colors = ["#AAAAAA", "#F4A261", "#E76F51"]  # grey, warm orange, coral-red
my_colors       = ["#06D6A0", "#118AB2", "#FFD166", "#EF476F"]  # teal, blue, yellow, pink

def plot_segments(ax, xs, ys, color, linewidth, linestyle, alpha, label):
    """Plot segments split on x resets (x==0) to avoid spurious connecting lines."""
    seg_x, seg_y = [], []
    first = True
    for x, y in zip(xs, ys):
        if x == 0:
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                        linestyle=linestyle, alpha=alpha,
                        label=label if first else "_nolegend_")
                first = False
            seg_x, seg_y = [], []
            continue
        seg_x.append(x)
        seg_y.append(y)
    if seg_x:
        ax.plot(seg_x, seg_y, color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha,
                label=label if first else "_nolegend_")

fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(16, 5.5))

for ax, xlabel, x_key in [
    (ax_step, "Training step",    "steps"),
    (ax_time, "Training time (s)", "times"),
]:
    for (label, path), color in zip(baselines.items(), baseline_colors):
        if not path.exists():
            print(f"Warning: baseline not found: {path}", file=sys.stderr)
            continue
        steps, losses, times = parse_log(path)
        if not steps:
            continue
        xs = steps if x_key == "steps" else times
        plot_segments(ax, xs, losses, color=color, linewidth=1.8,
                      linestyle="--", alpha=0.85, label=label)

    for i, run_path in enumerate(my_runs):
        color = my_colors[i % len(my_colors)]
        steps, losses, times = parse_log(run_path)
        if not steps:
            continue
        xs = steps if x_key == "steps" else times
        label = f"NS3+FTRL: {run_path.stem[:40]}"
        plot_segments(ax, xs, losses, color=color, linewidth=2.4,
                      linestyle="-", alpha=1.0, label=label)

    ax.set_ylim(3.2, 5.0)
    ax.axhline(3.28, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
            3.295, "Target (3.28)", fontsize=8, color="black", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Validation loss", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax_step.set_title("Val loss vs step", fontsize=13, fontweight="bold")
ax_time.set_title("Val loss vs time", fontsize=13, fontweight="bold")
ax_step.legend(fontsize=8, loc="upper right")
ax_time.legend(fontsize=8, loc="upper right")

out = HERE / "figure.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
