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

# Baselines log time in seconds: train_time:477.705s
# Our runs log time in milliseconds: train_time:323655ms
LOG_PATTERN_S  = re.compile(r'^step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+train_time:([0-9.]+)s\b')
LOG_PATTERN_MS = re.compile(r'^step:(\d+)/\d+\s+val_loss:([0-9.]+)\s+train_time:([0-9.]+)ms\b')

def parse_log(path):
    steps, losses, times = [], [], []
    for line in Path(path).read_text().splitlines():
        m = LOG_PATTERN_S.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            times.append(float(m.group(3)))          # already seconds
            continue
        m = LOG_PATTERN_MS.match(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            times.append(float(m.group(3)) / 1000.0)  # ms → seconds
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

VALID_RUNS = {
    "ns3_pure.txt":          "NS3 (3 iters, no FTRL)",
    "ns3ftrl_eta0p05.txt":   "NS3+FTRL (eta=0.05)",
    "ns3ftrl_switch300.txt": "NS3+FTRL→Muon (switch@300)",
}

runs_dir = HERE / "runs"
my_runs = [runs_dir / name for name in VALID_RUNS if (runs_dir / name).exists()]

def run_label(path):
    final = next((l for l in path.read_text().splitlines()[::-1] if re.match(r'^step:\d+/\d+\s+val_loss:', l)), "")
    val = re.search(r'val_loss:([0-9.]+)', final)
    last_step = re.search(r'^step:(\d+)/(\d+)', final)
    suffix = f"  [{last_step.group(1)}/{last_step.group(2)} steps, loss {val.group(1)}]" if val and last_step else ""
    base = VALID_RUNS.get(path.name, path.stem)
    return f"{base}{suffix}"

baseline_colors = ["#AAAAAA", "#F4A261", "#E76F51"]   # grey, amber, coral
my_colors       = ["#00B4D8", "#FF006E", "#AAFF00"]   # sky blue, hot pink, lime

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

import math as _math
import numpy as np

_ETA0       = 0.3
_ETA_LAMBDA = _math.log(3.0) / 100.0
_SWITCH_STEP = 300
_TRAIN_STEPS = 3350

def eta_schedule(step):
    if step >= _SWITCH_STEP:
        return 0.0
    return _ETA0 * _math.exp(-_ETA_LAMBDA * step)

# Layout: 2 rows — top row has val-loss panels, bottom-left has eta schedule
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.5], hspace=0.45, wspace=0.3)
ax_step = fig.add_subplot(gs[0, 0])
ax_time = fig.add_subplot(gs[0, 1])
ax_eta  = fig.add_subplot(gs[1, 0])

for ax, xlabel, x_key in [
    (ax_step, "Training step",     "steps"),
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
        label = run_label(run_path)
        steps, losses, times = parse_log(run_path)
        if not steps:
            continue
        xs = steps if x_key == "steps" else times
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

# --- eta schedule subplot ---
steps_arr = np.arange(0, _TRAIN_STEPS + 1)
eta_arr   = np.array([eta_schedule(s) for s in steps_arr])

# FTRL phase
mask_ftrl = steps_arr < _SWITCH_STEP
ax_eta.fill_between(steps_arr[mask_ftrl], eta_arr[mask_ftrl], alpha=0.15, color="#AAFF00")
ax_eta.plot(steps_arr[mask_ftrl], eta_arr[mask_ftrl], color="#AAFF00", linewidth=2.0)

# Muon phase (eta=0)
mask_muon = steps_arr >= _SWITCH_STEP
ax_eta.fill_between(steps_arr[mask_muon], eta_arr[mask_muon], alpha=0.08, color="#00B4D8")
ax_eta.plot(steps_arr[mask_muon], np.zeros(mask_muon.sum()), color="#00B4D8", linewidth=2.0)

# Switch annotation
ax_eta.axvline(_SWITCH_STEP, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
ax_eta.text(_SWITCH_STEP + 30, _ETA0 * 0.55, f"switch → pure Muon\n(step {_SWITCH_STEP})",
            fontsize=8, color="white", alpha=0.85, va="center")

ax_eta.set_xlim(0, _TRAIN_STEPS)
ax_eta.set_ylim(-0.01, _ETA0 * 1.15)
ax_eta.set_xlabel("Training step", fontsize=10)
ax_eta.set_ylabel("FTRL eta (η)", fontsize=10)
ax_eta.set_title("η schedule: NS3+FTRL → Muon", fontsize=11, fontweight="bold")
ax_eta.spines["top"].set_visible(False)
ax_eta.spines["right"].set_visible(False)

# bottom-right: leave empty or add a note
fig.add_subplot(gs[1, 1]).set_visible(False)

out = HERE / "figure.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
