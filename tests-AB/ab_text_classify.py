"""Logit Text Classify's default question, to decide which wording picks the medium a prompt asks for.

Each wording goes out as the node's request, one choice question per prompt with the prompt as the state and
nodes.CLASSIFY_OPTIONS as the options, over the 72 render prompts. A prompt's medium is its render's manifest style.
Two conditions: "named" sends each prompt as written. "cues" drops the
prompt's leading medium phrase, so the choice reads only the rest, such as "halftone shading" or "16-bit".
Its output sets nodes.TEXT_CLASSIFY_QUESTION. Every interval is a 95% bootstrap over the prompts.
Writes data/text-classify/<run>/results.json with every record, the CLIP file name and the library version.
Usage: python tests-AB/ab_text_classify.py [--run NAME] [--clip FILE] [--limit N] [--seed N]
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import ab_env
from ab_question_shapes import bootstrap, interval, mean
from ab_release_gates import estimate, load_renders
from ab_style import names_medium, results_path, write_results
from ab_tagger import DEFAULT_CLIP
from ab_tagger_compare import DEFAULT_MANIFEST

TEXT_CLASSIFY_ROOT = Path(__file__).resolve().parent / "data" / "text-classify"
HARNESS_NAME = "ab_text_classify"
WORDINGS = (
    "What art style does this prompt ask for?",
    "What is the art style of the image this prompt describes?",
    "What medium does this prompt ask for?",
)
CONDITION_NAMED = "named"
CONDITION_CUES = "cues"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--run", default=None, help="the run folder's name under data/text-classify, default <date>")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N render prompts")
    parser.add_argument("--seed", type=int, default=7, help="seed for the bootstrap resamples")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be 1 or more, got {args.limit}")
    return args


def cue_text(prompt: str, medium: str) -> str:
    """The prompt without its leading medium phrase, the words before its first " of " or comma, when one of those
    words shares a form with a word of the medium name. Any other prompt comes back as written."""
    from logit_classifier.toolkit.tags import in_prompt, prompt_word_forms

    cut = min((index for index in (prompt.find(" of "), prompt.find(",")) if index > 0), default=-1)
    head_forms = prompt_word_forms(prompt[:cut]) if cut > 0 else frozenset()

    if not any(in_prompt(word, head_forms) for word in medium.split()):
        return prompt
    return prompt[cut:].removeprefix(" of ").removeprefix(",").strip()


def accuracy(items: list[dict]) -> float:
    """The fraction of items whose choice is their medium."""
    return mean([item["choice"] == item["medium"] for item in items])


def difference_line(items: list[dict], wording: str, baseline: str, seed: int) -> str:
    """The wording's accuracy minus the baseline's over the same prompts, with its interval over prompt resamples."""
    right = {(item["unit"], item["wording"]): item["choice"] == item["medium"] for item in items}
    units = sorted({item["unit"] for item in items})
    differences = [right[(unit, wording)] - right[(unit, baseline)] for unit in units]
    spread = interval(bootstrap(lambda indices: mean([differences[index] for index in indices]), len(units), seed))

    return f"  minus {baseline!r} {mean(differences):+.4f} {spread}"


def wording_lines(items: list[dict], wordings: tuple[str, ...], seed: int) -> list[str]:
    """Each wording's accuracy with its interval over the prompts, and its paired difference from the first wording."""
    rows: list[dict] = []
    right = 0
    lines: list[str] = []

    for index, wording in enumerate(wordings):
        rows = [item for item in items if item["wording"] == wording]
        right = sum(item["choice"] == item["medium"] for item in rows)
        lines += [wording, f"  accuracy {estimate(accuracy, rows, seed)} ({right}/{len(rows)})"]
        if index > 0:
            lines.append(difference_line(items, wording, wordings[0], seed))
    return lines


def main() -> None:
    args = parse_args()
    out_path = results_path(TEXT_CLASSIFY_ROOT, args.run or date.today().isoformat())
    renders = load_renders(DEFAULT_MANIFEST)[: args.limit]
    options: list[str] = []
    unknown: list[str] = []
    toolkit: Any = None
    clip: Any = None
    classifier: Any = None
    texts: dict[str, dict[str, str]] = {}
    choice = ""
    confidence = 0.0
    items: list[dict] = []
    lines: list[str] = []

    ab_env.bootstrap()
    import logit_classifier

    from logit_tagger import engine, nodes

    options = nodes._classify_options(nodes.CLASSIFY_OPTIONS, nodes.TEXT_CLASSIFY_NAME)
    unknown = sorted({render.style for render in renders} - set(options))
    # Checked before the CLIP loads, since a medium outside the options can never be picked.
    if unknown:
        raise SystemExit(f"manifest styles {unknown} are not among nodes.CLASSIFY_OPTIONS {options}")
    texts = {CONDITION_NAMED: {render.id: render.prompt for render in renders},
             CONDITION_CUES: {render.id: cue_text(render.prompt, render.style) for render in renders}}
    toolkit = engine.toolkit(HARNESS_NAME)
    clip = ab_env.load_clip(args.clip)
    classifier = engine.build_classifier(clip, HARNESS_NAME, needs_images=False)
    with toolkit.skip_resident_loads(clip):
        for condition, text_by_id in texts.items():
            for wording in WORDINGS:
                for render in renders:
                    choice, confidence = nodes._text_choice(classifier, wording, options, text_by_id[render.id])
                    items.append({"unit": render.id, "medium": render.style, "condition": condition,
                                  "text": text_by_id[render.id], "wording": wording, "choice": choice,
                                  "confidence": float(confidence)})

    for condition, text_by_id in texts.items():
        lines += [f"{condition}: {sum(names_medium(text_by_id[render.id], render.style) for render in renders)}"
                  f"/{len(renders)} texts name their medium",
                  *wording_lines([item for item in items if item["condition"] == condition], WORDINGS, args.seed)]
    print(*lines, sep="\n")
    write_results(out_path, {"clip": args.clip, "library_version": logit_classifier.__version__, "options": options,
                             "wordings": list(WORDINGS), "records": items})
    print(f"raw results: {out_path}")


if __name__ == "__main__":
    main()
