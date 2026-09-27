"""Turn a training run's JSON into a figure and a plain-language summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"

COLORS = {"mb": "#c2410c", "readout": "#1d4ed8"}
LABELS = {"mb": "mushroom body (dopamine-gated KC→MBON)",
          "readout": "trained linear readout (control)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args()

    res = json.loads((OUT / f"{args.tag}.json").read_text())
    base = res["random_baseline"]["mean_distance"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.axhline(base, color="#6b7280", ls="--", lw=1.4,
               label=f"random policy ({base:.1f})")
    for kind, curve in res["curves"].items():
        if not curve:
            continue
        x = [c["episode"] for c in curve]
        y = [c["eval_distance"] for c in curve]
        ax.plot(x, y, "-o", ms=3.5, lw=1.8, color=COLORS.get(kind, "k"),
                label=LABELS.get(kind, kind))
    ax.set_xlabel("training episode")
    ax.set_ylabel("evaluation distance (greedy policy)")
    ax.set_title("Learning on the runner")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    mb = res["curves"].get("mb") or []
    if mb and "weights" in mb[0]:
        x = [c["episode"] for c in mb]
        dep = [c["weights"].get("frac_depressed", 0) for c in mb]
        pot = [c["weights"].get("frac_potentiated", 0) for c in mb]
        mag = [c["weights"].get("mean_abs_change", 0) for c in mb]
        ax.plot(x, dep, "-o", ms=3.5, color="#0e7490", label="fraction depressed")
        ax.plot(x, pot, "-o", ms=3.5, color="#be123c", label="fraction potentiated")
        ax.set_ylabel("fraction of plastic synapses")
        ax.set_xlabel("training episode")
        ax2 = ax.twinx()
        ax2.plot(x, mag, "-s", ms=3.0, color="#6b7280", lw=1.2,
                 label="mean |Δw|")
        ax2.set_ylabel("mean |Δw|", color="#6b7280")
        ax2.spines[["top"]].set_visible(False)
        n = mb[0]["weights"].get("n_plastic", 0)
        ax.set_title(f"Mushroom body plasticity ({n:,} plastic synapses)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")
    ax.spines[["top"]].set_visible(False)

    fig.tight_layout()
    png = OUT / f"{args.tag}.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")

    print(f"\nrandom-policy floor: {base:.1f}")
    cal = res.get("calibration", {})
    if cal:
        print(f"network: {cal.get('mean_hz', 0):.2f} Hz mean rate, "
              f"synchrony {cal.get('synchrony', 0):.2f}, "
              f"{cal.get('active_frac', 0):.1%} active")
    print(f"\n{'agent':10s} {'first':>8s} {'best':>8s} {'final':>8s} {'vs random':>10s}")
    for kind, curve in res["curves"].items():
        if not curve:
            continue
        ys = [c["eval_distance"] for c in curve]
        print(f"{kind:10s} {ys[0]:8.1f} {max(ys):8.1f} {ys[-1]:8.1f} "
              f"{max(ys) / base:9.2f}x")


if __name__ == "__main__":
    main()
