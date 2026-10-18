"""Compare the library's packed scorer against its per-branch generate path on real images.

Both paths answer the same noul questions about the same resized image. The packed path must
flip no decision at the threshold and stay within MAX_DIFF of the per-branch path.
Usage: python tests-AB/ab_packed_parity.py [--clip FILE]
"""

import argparse
import time
from pathlib import Path

import ab_env

RENDERS = Path(__file__).resolve().parent / "data" / "release-gates" / "renders"
SAMPLES = Path(r"C:\Users\Blake\Documents\ComfyUI\custom_nodes\ComfyUI-ContextAnchoredTileRefine\samples")
ABSENT = ["a giraffe", "a bicycle", "a bowl of soup", "a laptop computer", "snow"]
# Each render's tags are its intended things, every one named by its prompt under the toolkit's in_prompt
# and judged present in renders/labels.json.
CASES = [
    (RENDERS / "g52.png",
     ["market", "vendors", "fruit", "wooden carts", "shoppers", "baskets", "awnings", "dog"]),
    (SAMPLES / "dragon-4x-vl-upscale-preview.jpg",
     ["dragon", "bell", "ships", "castle", "mountains", "banners", "people", "sea", "sky", "armor"]),
    (SAMPLES / "cyberpunk-city-4k-preview.jpg",
     ["skyscrapers", "neon signs", "bridge", "clouds", "street signs", "abstract art", "windows"]),
    (RENDERS / "g57.png",
     ["red barn", "farmer", "crops", "chickens", "brown horse", "fence", "blue wheelbarrow", "gate"]),
]
# The prototype's largest difference was 0.0032, so this leaves room without hiding a defect.
MAX_DIFF = 0.01


def run(classifier, backend, method, request, image):
    import torch

    backend.score = method
    torch.cuda.synchronize()
    started = time.perf_counter()
    response, _diagnostics = classifier.classify(request, image=image)
    torch.cuda.synchronize()
    return response, (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", default="qwen3vl_4b_fp8_scaled.safetensors")
    args = parser.parse_args()
    ab_env.bootstrap()

    from ab_tagger import load_image
    from logit_classifier import Classifier, Config, NoulQuestion, SystemOneRequest
    from logit_classifier.backends.comfy_clip import ComfyClipBackend
    from logit_classifier.toolkit.comfyui import fit_image

    from logit_tagger.engine import VL_MAX_PIXELS
    from logit_tagger.tagging import tag_settings

    settings = tag_settings()
    threshold = settings.verify_thresholds.proposed
    clip = ab_env.load_clip(args.clip)
    backend = ComfyClipBackend(clip)
    classifier = Classifier(Config(use_prior_debias=False, calibration_path=None), backend=backend)
    worst = 0.0
    flips = 0

    for path, tags in CASES:
        image = fit_image(load_image(path), VL_MAX_PIXELS)
        questions = {f"q{i}": NoulQuestion(instructions=settings.verify_statement.replace("{TAG}", tag))
                     for i, tag in enumerate(tags + ABSENT)}
        request = SystemOneRequest(state="", questions=questions)
        slow, slow_ms = run(classifier, backend, backend._score_by_generate, request, image)
        fast, fast_ms = run(classifier, backend, backend._score_packed, request, image)
        print(f"== {path.name} {tuple(image.shape)} {len(questions)} questions: "
              f"per-branch {slow_ms:.0f} ms, packed {fast_ms:.0f} ms, {slow_ms / fast_ms:.1f}x")
        for i, tag in enumerate(tags + ABSENT):
            a = slow.answers[f"q{i}"].noul
            b = fast.answers[f"q{i}"].noul
            flipped = (a >= threshold) != (b >= threshold)
            worst = max(worst, abs(a - b))
            flips += flipped
            print(f"  {a:.6f} {b:.6f} {'FLIP' if flipped else '    '} {tag}")
    del backend.score
    print(f"\nlargest |diff| {worst:.6f} (limit {MAX_DIFF}), flips at {threshold}: {flips}")
    return 0 if flips == 0 and worst < MAX_DIFF else 1


if __name__ == "__main__":
    raise SystemExit(main())
