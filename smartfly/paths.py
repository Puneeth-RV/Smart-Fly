"""Where things live on disk. Everything is relative to the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"  # the MaleCNS release files (downloaded)
PROCESSED = DATA / "processed"  # the whole connectome as one matrix (built)
VISUAL_SYSTEM = DATA / "visual_system.npz"  # just the eye and optic lobe (built)

MODELS = ROOT / "models"  # trained flies, committed
RESULTS = ROOT / "results"  # measurements quoted in the README, committed
RUNS = ROOT / "runs"  # training output, not committed
SITE = ROOT / "site"  # the web page

CAMPAIGN = RUNS / "campaign"
