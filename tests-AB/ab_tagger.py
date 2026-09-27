"""Run the Logit Tagger pipeline on image files over a real Qwen3-VL-4B CLIP, with no server.

The output sets the backend constants PROPOSE_INSTRUCTION, PROMPT_ANCHOR and NOUL_THRESHOLD.
Usage: python tests-AB/ab_tagger.py <image> [<image> ...] [--prompt TEXT] [--threshold P]
"""

import argparse
from pathlib import Path

import ab_env

DEFAULT_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"


def parse_args(threshold_default: float) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--prompt", default="", help="text for the prompt input, empty by default")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--threshold", type=float, default=threshold_default,
                        help=f"noul threshold, default the module's NOUL_THRESHOLD {threshold_default}")
    parser.add_argument("--instruction", default=None, help="replaces PROPOSE_INSTRUCTION for this run")
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


def report(path: Path, detail, threshold: float) -> str:
    kept = set(detail.tags.split(", ")) if detail.tags else set()
    ranked = sorted(detail.candidates, key=lambda pair: pair[1], reverse=True)
    per_candidate_ms = detail.verify_ms / len(ranked) if ranked else 0.0
    lines = [
        f"== {path}",
        f"size: {detail.size_in[1]}x{detail.size_in[0]} -> {detail.size_out[1]}x{detail.size_out[0]} (width x height)",
        "proposal:",
        detail.proposal,
        f"candidates: {len(ranked)}, * marks a tag at or above the threshold {threshold}",
    ]

    lines += [f"  {'*' if tag in kept else ' '} {probability:.4f}  {tag}" for tag, probability in ranked]
    lines += [
        f"kept: {detail.tags}",
        f"propose {detail.propose_ms:.0f} ms, verify {detail.verify_ms:.0f} ms, "
        f"verify per candidate {per_candidate_ms:.1f} ms",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ab_env.bootstrap()
    from logit_tagger import tagging

    args = parse_args(tagging.NOUL_THRESHOLD)
    missing = [str(path) for path in args.images if not path.is_file()]
    clip = None
    classifier = None
    detail = None

    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")

    clip = ab_env.load_clip(args.clip)
    # One classifier across images, as tag_images builds one per batch.
    classifier = tagging.build_classifier(clip)
    for path in args.images:
        detail = tagging.tag_image(clip, load_image(path), args.prompt, threshold=args.threshold,
                                   instruction=args.instruction, classifier=classifier)
        print(report(path, detail, args.threshold), flush=True)


if __name__ == "__main__":
    main()
