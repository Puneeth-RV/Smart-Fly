"""Render the README animation: the fly playing a level, with what it sees.

Left: the game, with a glow where the fly was looking when it chose its button.
Right: the retina its photoreceptors receive, their response, and the buttons it
weighed. Plays the trained fly's top choice from random starts until one attempt
reaches the end, and records that attempt.

    python scripts/make_gif.py models/fly-campaign.pt --level 1-1 --out docs/media/fly-1-1.gif
"""

from __future__ import annotations

import argparse
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
SCALE = 1.5  # game pixels -> GIF pixels
PANEL_W = 330


def font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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

    d = ImageDraw.Draw(img)
    f_head, f_small = font(15), font(12)
    px, w = gw + 18, PANEL_W - 34
    y = 14
    d.text((px, y), f"Level {level}  ·  distance {x:,}", fill=INK, font=f_head)
    y += 30
    half = (w - 10) // 2
    for i, (title, values) in enumerate(
        (("Reaches its eye", retina), ("Its photoreceptors", photo))
    ):
        x0 = px + i * (half + 10)
        d.text((x0, y), title, fill=DIM, font=f_small)
        img.paste(grey(values, RETINA, half), (x0, y + 18))
    y += 18 + round(half * RETINA[0] / RETINA[1]) + 16
    d.text((px, y), "Its choice", fill=DIM, font=f_small)
    y += 20
    for i, name in enumerate(ACTIONS):
        on = i == action
        d.text((px, y), LABELS[name], fill=INK if on else DIM, font=f_small)
        bx, bw = px + 100, w - 100
        d.rectangle((bx, y + 5, bx + bw, y + 10), fill=PANEL)
        d.rectangle((bx, y + 5, bx + max(1, int(bw * probs[i])), y + 10), fill=HEAT if on else DIM)
        y += 19
    d.text((px, gh - 22), "Smart-Fly · 92,743 simulated neurons", fill=DIM, font=f_small)
    return img


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
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=67, loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
