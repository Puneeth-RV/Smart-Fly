"""Fetch the three MaleCNS v1.0 tables the build needs (~1.15 GB).

The release is published by Google Research and HHMI Janelia under CC-BY 4.0;
see https://male-cns.janelia.org/download/. The other files in the release are
for spatial analysis and are not needed here.

    python -m smartfly.connectome.download
"""

from __future__ import annotations

import shutil
import urllib.request

from smartfly.paths import RAW

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have {name}")
            continue
        print(f"fetching {name} ...", flush=True)
        part = dest.with_suffix(".part")
        with urllib.request.urlopen(f"{BASE}/{name}") as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        part.replace(dest)
    print(f"done: {RAW}")


if __name__ == "__main__":
    main()
