"""Render the README animation: the fly playing a level, with what it sees.

Left: the game, with a glow where the fly was looking when it chose its button.
Right: the retina its photoreceptors receive, their response, and the buttons it
weighed. Plays the trained fly's top choice from random starts until one attempt
reaches the end, and records that attempt.

    python scripts/make_gif.py models/fly-campaign.pt --level 1-4 --out docs/media/fly-1-4.gif
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from smartfly.game import ACTIONS, PLAYFIELD, RETINA, Mario
from smartfly.policy import Player

BG, PANEL, INK, DIM, HEAT, ACCENT = (
    (17, 19, 22),
    (24, 27, 31),
    (233, 230, 223),
    (128, 132, 139),
    (233, 164, 70),
    (124, 196, 155),
)
LABELS = {
    "none": "Nothing",
    "right": "Right",
    "run-jump": "Right + jump",
    "run": "Right + run",
    "run-jump-b": "Running jump",
    "jump": "Jump",
    "left": "Left",
}
SCALE = 2  # game pixels -> GIF pixels; a whole number keeps every NES pixel square
PANEL_W = 330


def font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def grey(values: np.ndarray, shape: tuple[int, int], width: int) -> Image.Image:
    v = values.reshape(shape)
    lo, hi = np.percentile(v, 5), np.percentile(v, 98)
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (np.array(PANEL) + (np.array(INK) - np.array(PANEL)) * v[..., None]).astype(np.uint8)
    h = round(width * shape[0] / shape[1])
    return Image.fromarray(rgb).resize((width, h), Image.NEAREST)


def render(frame, attention, retina, photo, probs, action, level, x) -> Image.Image:
    gw, gh = int(256 * SCALE), int(240 * SCALE)
    img = Image.new("RGB", (gw + PANEL_W, gh), BG)

    game = Image.fromarray(frame.astype(np.uint8)).resize((gw, gh), Image.NEAREST)
    if attention is not None:
        a = attention.reshape(16, 20)
        a = np.clip((a - 0.25) / 0.75, 0, 1) ** 2
        top, bottom = int(PLAYFIELD.start * SCALE), int(PLAYFIELD.stop * SCALE)
        glow = Image.fromarray((a * 170).astype(np.uint8)).resize((gw, bottom - top), Image.BICUBIC)
        layer = Image.new("RGB", (gw, bottom - top), HEAT)
        region = game.crop((0, top, gw, bottom))
        game.paste(Image.composite(layer, region, glow), (0, top))
    img.paste(game, (0, 0))

    # The side panel is drawn at half size with unsmoothed text, then doubled like the
    # game's pixels, so it stays sharp when GitHub and Retina screens rescale the GIF.
    pw, ph = PANEL_W // 2, gh // 2
    panel = Image.new("RGB", (pw, ph), BG)
    d = ImageDraw.Draw(panel)
    d.fontmode = "1"
    f = font(9)
    px, w = 9, pw - 17
    y = 7
    d.text((px, y), f"{level} · distance {x:,}", fill=INK, font=f)
    y += 17
    half = (w - 5) // 2
    for i, (title, values) in enumerate((("Its eye", retina), ("Receptors", photo))):
        x0 = px + i * (half + 5)
        d.text((x0, y), title, fill=DIM, font=f)
        panel.paste(grey(values, RETINA, half), (x0, y + 12))
    y += 12 + round(half * RETINA[0] / RETINA[1]) + 12
    d.text((px, y), "Its choice", fill=DIM, font=f)
    y += 14
    for i, name in enumerate(ACTIONS):
        on = i == action
        d.text((px, y), LABELS[name], fill=INK if on else DIM, font=f)
        bx, bw = px + 78, w - 78
        d.rectangle((bx, y + 4, bx + bw, y + 6), fill=PANEL)
        d.rectangle((bx, y + 4, bx + max(1, int(bw * probs[i])), y + 6), fill=HEAT if on else DIM)
        y += 13
    d.text((px, ph - 14), "92,743 simulated neurons", fill=DIM, font=f)
    img.paste(panel.resize((pw * 2, ph * 2), Image.NEAREST), (gw, 0))
    return img


def shrink(path: Path) -> None:
    """Re-encode with ffmpeg, if installed, so each frame stores only what changed."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".tmp.gif")
    graph = (
        "[0:v]split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-filter_complex", graph, str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--out", default="docs/media/fly.gif")
    ap.add_argument("--tries", type=int, default=10)
    args = ap.parse_args()
    player = Player(args.checkpoint)
    game = Mario(args.level)
    rng = np.random.default_rng(1)
    for attempt in range(args.tries):
        retina = game.reset()
        for _ in range(int(rng.integers(0, 31))):
            retina, *_ = game.step(0)
        player.reset()
        frames, flag, info = [], False, {}
        while not game.done and len(frames) < 1500:
            dec = player.decide(retina, attention=True)
            frames.append(
                render(
                    game.frame,
                    dec.attention,
                    retina,
                    dec.features[player.fly.layout["eye"]],
                    dec.probs,
                    dec.action,
                    args.level,
                    int(game.max_x),
                )
            )
            for _ in range(player.frame_skip):
                retina, _, done, info = game.step(dec.action)
                flag |= bool(info.get("flag_get"))
                if done:
                    break
        print(f"attempt {attempt + 1}: distance {game.max_x}{', reached the end' if flag else ''}")
        if flag:
            break
    else:
        raise SystemExit("no attempt reached the end; not writing a GIF of a failure")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames += [frames[-1]] * 20  # hold the end for a moment
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=67, loop=0, optimize=True)
    shrink(out)
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
