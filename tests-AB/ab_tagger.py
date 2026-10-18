"""Run the Logit Tagger pipeline on image files over a real Qwen3-VL-4B CLIP, with no server.

The output sets the TagSettings fields propose_instruction and verify_thresholds.
Usage: python tests-AB/ab_tagger.py <image> [<image> ...] [--prompt TEXT] [--instruction TEXT] [--keep-non-things]
       [--positions]
"""

import argparse
from pathlib import Path

import ab_env

DEFAULT_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--prompt", default="", help="text for the prompt input, empty by default")
    parser.add_argument("--instruction", default="",
                        help="Logit Custom Tagger's instruction, empty for the default wording")
    parser.add_argument("--keep-non-things", action="store_true",
                        help="Logit Custom Tagger's things_only off, so the thing check judges only the prompt's tags")
    parser.add_argument("--positions", action="store_true",
                        help="the positions switch on, so each kept tag is placed on the image's strips")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
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
    if trace.terms:
        lines += ["positions:"]
        lines += [f"  {tag}: {term or '-'}" for tag, term in zip(trace.kept, trace.terms, strict=True)]
        lines += [f"unplaced: {', '.join(trace.unplaced) or '-'}"]
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
    settings = None
    clip = None
    traces = []

    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")

    # The default instruction with things_only on builds the Tagger's own settings.
    settings = tagging.custom_settings(args.instruction, strict=False, things_only=not args.keep_non_things)
    clip = ab_env.load_clip(args.clip)
    traces = tagging.tag_images(clip, [load_image(path) for path in args.images], [args.prompt] * len(args.images),
                                settings=settings, positions=args.positions)
    for path, trace in zip(args.images, traces, strict=True):
        print(report(path, trace), flush=True)


if __name__ == "__main__":
    main()
