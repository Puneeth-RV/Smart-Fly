"""Inline brain.js and game.js into the published page.

They stay separate files in the repo for editing, but ship inlined: the artifact
CSP admits external scripts only from a fixed CDN list, and inlining removes any
question about whether sibling files count.
"""
from pathlib import Path

here = Path(__file__).parent
src = (here / "index.src.html").read_text()
for name in ("brain.js", "game.js"):
    src = src.replace(f"/*INLINE:{name}*/", (here / name).read_text())
assert "/*INLINE:" not in src, "an inline marker was left unresolved"
(here / "index.html").write_text(src)
print(f"built index.html ({len(src)/1024:.0f} KB)")
