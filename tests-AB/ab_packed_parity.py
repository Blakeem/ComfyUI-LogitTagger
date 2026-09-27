"""Compare the library's packed scorer against its per-branch generate path on real images.

Both paths answer the same noul questions about the same resized image. The packed path must
flip no decision at the threshold and stay within MAX_DIFF of the per-branch path.
Usage: python tests-AB/ab_packed_parity.py [--clip FILE]
"""

import argparse
import time
from pathlib import Path

import ab_env

INPUT = Path(r"C:\Users\Blake\Documents\ComfyUI\input")
SAMPLES = Path(r"C:\Users\Blake\Documents\ComfyUI\custom_nodes\ComfyUI-ContextAnchoredTileRefine\samples")
ABSENT = ["a giraffe", "a bicycle", "a bowl of soup", "a laptop computer", "snow"]
CASES = [
    (INPUT / "krea2-small-input.png",
     ["woman", "red flowers", "large red moon", "stone arches", "ivy", "dark forest", "black clothing",
      "red hair", "white wolf", "glowing lantern", "silver sword", "mossy trees", "stars", "night sky"]),
    (SAMPLES / "dragon-4x-vl-upscale-preview.jpg",
     ["dragon", "bell", "ships", "castle", "mountains", "banners", "people", "sea", "sky", "armor"]),
    (SAMPLES / "cyberpunk-city-4k-preview.jpg",
     ["skyscrapers", "neon signs", "bridge", "clouds", "street signs", "abstract art", "windows"]),
    (INPUT / "bb3475fda1707a0253f3cb48afbf1873-cc_ft_576.webp",
     ["house", "tree", "car", "lawn", "chimney", "fence", "swimming pool", "sidewalk"]),
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

    from logit_tagger.engine import resize_for_vl
    from logit_tagger.tagging import VERIFY_STATEMENT, VERIFY_THRESHOLD

    clip = ab_env.load_clip(args.clip)
    backend = ComfyClipBackend(clip)
    classifier = Classifier(Config(use_prior_debias=False, calibration_path=None), backend=backend)
    worst = 0.0
    flips = 0

    for path, tags in CASES:
        image = resize_for_vl(load_image(path))
        questions = {f"q{i}": NoulQuestion(instructions=VERIFY_STATEMENT.replace("{TAG}", tag))
                     for i, tag in enumerate(tags + ABSENT)}
        request = SystemOneRequest(state="", questions=questions)
        slow, slow_ms = run(classifier, backend, backend._score_by_generate, request, image)
        fast, fast_ms = run(classifier, backend, backend._score_packed, request, image)
        print(f"== {path.name} {tuple(image.shape)} {len(questions)} questions: "
              f"per-branch {slow_ms:.0f} ms, packed {fast_ms:.0f} ms, {slow_ms / fast_ms:.1f}x")
        for i, tag in enumerate(tags + ABSENT):
            a = slow.answers[f"q{i}"].noul
            b = fast.answers[f"q{i}"].noul
            flipped = (a >= VERIFY_THRESHOLD) != (b >= VERIFY_THRESHOLD)
            worst = max(worst, abs(a - b))
            flips += flipped
            print(f"  {a:.6f} {b:.6f} {'FLIP' if flipped else '    '} {tag}")
    del backend.score
    print(f"\nlargest |diff| {worst:.6f} (limit {MAX_DIFF}), flips at {VERIFY_THRESHOLD}: {flips}")
    return 0 if flips == 0 and worst < MAX_DIFF else 1


if __name__ == "__main__":
    raise SystemExit(main())
