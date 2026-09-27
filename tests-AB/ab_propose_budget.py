"""Compare PROPOSE_MAX_TOKENS budgets on the same images over a real Qwen3-VL-4B CLIP, with no server.

The output sets the backend constant PROPOSE_MAX_TOKENS.
Usage: python tests-AB/ab_propose_budget.py <image> [<image> ...] [--budgets 128,192]
"""

import argparse
from pathlib import Path

import ab_env
from ab_tagger import DEFAULT_CLIP, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--budgets", default="128,192", help="comma separated token budgets, the first is the baseline")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    return parser.parse_args()


def count_items(proposal: str) -> int:
    """Items the proposal wrote before parse_candidates removed repeats, so a loop shows as items above unique."""
    from logit_classifier.tags import complete_tags

    # The appended separator counts the last item, which complete_tags would treat as mid tag.
    return len(complete_tags(proposal + ","))


def main() -> None:
    ab_env.bootstrap()
    from logit_tagger import engine, tagging

    args = parse_args()
    budgets = [int(value) for value in args.budgets.split(",")]
    missing = [str(path) for path in args.images if not path.is_file()]
    token_counts: list[int] = []
    clip = None
    classifier = None
    generate = None

    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")

    clip = ab_env.load_clip(args.clip)
    classifier = engine.build_classifier(clip, tagging.NODE_NAME)
    generate = clip.generate

    # Records how many tokens each decode wrote, which TagTrace does not carry.
    def counting_generate(*call_args, **call_kwargs):
        ids = generate(*call_args, **call_kwargs)
        token_counts.append(len(ids))
        return ids

    clip.generate = counting_generate
    # The first decode pays the model load, so it is run once and discarded.
    tagging.tag_image(clip, classifier, load_image(args.images[0]), known_things={})

    for path in args.images:
        image = load_image(path)
        runs = {}
        for budget in budgets:
            tagging.PROPOSE_MAX_TOKENS = budget
            trace = tagging.tag_image(clip, classifier, image, known_things={})
            runs[budget] = (trace, token_counts[-1])
        print(f"== {path.name}")
        for budget, (trace, tokens) in runs.items():
            kept = trace.tags.split(", ") if trace.tags else []
            print(f"  budget {budget}: {tokens} tokens{' (hit the budget)' if tokens >= budget else ''}, "
                  f"{count_items(trace.reply)} items, {len(trace.candidates)} unique, "
                  f"{len(kept)} kept, propose {trace.propose_ms:.0f} ms, verify {trace.verify_ms:.0f} ms")
        base_trace = runs[budgets[0]][0]
        base_kept = set(base_trace.tags.split(", ")) if base_trace.tags else set()
        for budget in budgets[1:]:
            trace = runs[budget][0]
            scores = dict(zip(trace.candidates, trace.scores, strict=True))
            kept = set(trace.tags.split(", ")) if trace.tags else set()
            gained = sorted(kept - base_kept, key=lambda tag: -scores[tag])
            lost = sorted(base_kept - kept)
            print(f"  gained at {budget}: " + (", ".join(f"{tag} {scores[tag]:.2f}" for tag in gained) or "none"))
            print(f"  lost at {budget}: " + (", ".join(lost) or "none"))
        print(flush=True)


if __name__ == "__main__":
    main()
