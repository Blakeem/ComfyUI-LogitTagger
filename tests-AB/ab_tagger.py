"""Run the Logit Tagger pipeline on image files over a real Qwen3-VL-4B CLIP, with no server.

The output sets the TagSettings fields propose_instruction, verify_threshold and prompt_only_threshold.
Usage: python tests-AB/ab_tagger.py <image> [<image> ...] [--prompt TEXT] [--instruction TEXT]
"""

import argparse
from pathlib import Path

import ab_env

DEFAULT_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--prompt", default="", help="text for the prompt input, empty by default")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--instruction", default=None, help="replaces the propose_instruction setting for this run")
    return parser.parse_args()


def load_image(path: Path):
    """LoadImage's pixels for one file: EXIF rotation applied, RGB, float in [0, 1], shape [1, H, W, 3]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels)[None,]


def report(path: Path, trace) -> str:
    kept = set(trace.kept)
    ranked = sorted(zip(trace.candidates, trace.origins, trace.scores, strict=True),
                    key=lambda row: row[2], reverse=True)
    lines = [
        f"== {path}",
        f"size: {trace.size_in[1]}x{trace.size_in[0]} -> {trace.size_out[1]}x{trace.size_out[0]} (width x height)",
        "reply:",
        trace.reply,
        f"candidates: {len(ranked)}, * marks a kept tag",
    ]

    lines += [f"  {'*' if tag in kept else ' '} {score:.4f}  {origin:6}  {tag}" for tag, origin, score in ranked]
    lines += [f"dropped: {len(trace.dropped)}"]
    lines += [f"  {name}: {reason}" for name, reason in trace.dropped]
    lines += [
        f"kept: {trace.tags}",
        f"propose {trace.propose_ms:.0f} ms, verify {trace.verify_ms:.0f} ms",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ab_env.bootstrap()
    from logit_tagger import tagging

    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    clip = None
    traces = []

    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")

    clip = ab_env.load_clip(args.clip)
    traces = tagging.tag_images(clip, [load_image(path) for path in args.images], [args.prompt] * len(args.images),
                                instruction=args.instruction)
    for path, trace in zip(args.images, traces, strict=True):
        print(report(path, trace), flush=True)


if __name__ == "__main__":
    main()
