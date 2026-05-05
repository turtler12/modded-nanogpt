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

LOG_PATTERN = re.compile(r'step:(\d+)/\d+\s+val_loss:([0-9.]+)')

def parse_log(path):
    steps, losses = [], []
    for line in Path(path).read_text().splitlines():
        m = LOG_PATTERN.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses

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

fig, ax = plt.subplots(figsize=(9, 5.5))

baseline_colors = ["#9E9E9E", "#607D8B", "#1565C0"]
my_colors = ["#E53935", "#FF6F00", "#2E7D32", "#6A1B9A"]

for (label, path), color in zip(baselines.items(), baseline_colors):
    if not path.exists():
        print(f"Warning: baseline not found: {path}", file=sys.stderr)
        continue
    steps, losses = parse_log(path)
    if not steps:
        continue
    # Only plot from step > 0 (skip the step:0 initialization)
    pairs = [(s, l) for s, l in zip(steps, losses) if s > 0]
    if not pairs:
        continue
    s, l = zip(*pairs)
    ax.plot(s, l, color=color, linewidth=1.5, linestyle="--", alpha=0.7, label=label)

for i, run_path in enumerate(my_runs):
    color = my_colors[i % len(my_colors)]
    steps, losses = parse_log(run_path)
    pairs = [(s, l) for s, l in zip(steps, losses) if s > 0]
    if not pairs:
        continue
    s, l = zip(*pairs)
    label = f"NS3+FTRL: {run_path.stem[:40]}"
    ax.plot(s, l, color=color, linewidth=2.0, label=label)

# Target line
ax.axhline(3.28, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.text(50, 3.282, "Target (3.28)", fontsize=8, color="black", alpha=0.6)

ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Validation loss", fontsize=11)
ax.set_title("NS3+FTRL vs SOTA baselines", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(top=min(6.0, ax.get_ylim()[1]))

out = HERE / "figure.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()
