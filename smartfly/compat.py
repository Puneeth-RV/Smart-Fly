"""Make nes-py 8.2.1 and gym-super-mario-bros 7.4.0 work under NumPy 2.

Both libraries predate NumPy 2, where arithmetic on a uint8 scalar no longer
widens: `ram[0x6d] * 0x100` raises instead of giving a position, and the ROM
header's `16 * header[4]` overflows on load. Each fix converts the byte to a
Python int first. They are applied to the installed files, once:

    python -m smartfly.compat

Re-running is harmless. `check()` is called when the game is created, so a
missing fix fails with this instruction rather than a cryptic overflow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# (file inside the package, original line, fixed line)
FIXES: list[tuple[str, str, str, str]] = [
    ("nes_py", "_rom.py", "return 16 * self.header[4]", "return int(16 * self.header[4])"),
    ("nes_py", "_rom.py", "return 8 * self.header[5]", "return int(8 * self.header[5])"),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x075f] * 4 + self.ram[0x075c]",
        "return int(self.ram[0x075f]) * 4 + int(self.ram[0x075c])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x075f] + 1",
        "return int(self.ram[0x075f]) + 1",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x075c] + 1",
        "return int(self.ram[0x075c]) + 1",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x0760] + 1",
        "return int(self.ram[0x0760]) + 1",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x075a]",
        "return int(self.ram[0x075a])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x6d] * 0x100 + self.ram[0x86]",
        "return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x03b8]",
        "return int(self.ram[0x03b8])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x00b5]",
        "return int(self.ram[0x00b5])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return _STATUS_MAP[self.ram[0x0756]]",
        "return _STATUS_MAP[int(self.ram[0x0756])]",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "return self.ram[0x000e]",
        "return int(self.ram[0x000e])",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "if self.ram[address] in _STAGE_OVER_ENEMIES:",
        "if int(self.ram[address]) in _STAGE_OVER_ENEMIES:",
    ),
    (
        "gym_super_mario_bros",
        "smb_env.py",
        "change_area_timer = self.ram[0x06DE]",
        "change_area_timer = int(self.ram[0x06DE])",
    ),
]


def _file(package: str, name: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError(f"{package} is not installed; run `pip install -e .` first")
    return Path(list(spec.submodule_search_locations)[0]) / name


def _missing() -> list[tuple[str, str, str, str]]:
    """Fixes not yet applied."""
    return [f for f in FIXES if f[3] not in _file(f[0], f[1]).read_text()]


def apply() -> int:
    """Apply every missing fix in place. Returns how many were applied."""
    todo = _missing()
    for package, name, old, new in todo:
        path = _file(package, name)
        text = path.read_text()
        if text.count(old) != 1:
            raise RuntimeError(f"{path}: expected exactly one `{old}`; is this the pinned version?")
        path.write_text(text.replace(old, new))
    return len(todo)


def check() -> None:
    """Raise with instructions if a fix is missing; also restore NumPy names old gym uses."""
    for old, new in (("bool8", np.bool_), ("float_", np.float64), ("int_", np.int64)):
        if not hasattr(np, old):
            setattr(np, old, new)
    if _missing():
        raise RuntimeError(
            "The NES emulator needs its NumPy 2 fixes: run `python -m smartfly.compat`"
        )


if __name__ == "__main__":
    n = apply()
    print(f"applied {n} fix{'es' if n != 1 else ''}" if n else "all fixes already applied")
    sys.exit(0)
