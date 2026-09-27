"""Compare PROPOSE_INSTRUCTION wordings on the same images over a real Qwen3-VL-4B CLIP, with no server.

The output sets the backend constant PROPOSE_INSTRUCTION. Each wording runs through the node's own
propose stage, so the repeat stop applies, and a reply it ended counts as a loop.
Usage: python tests-AB/ab_propose_prompts.py <image> [<image> ...]
"""

import argparse
from pathlib import Path

import ab_env
from ab_tagger import DEFAULT_CLIP, load_image

TAIL = ("Use short lowercase noun phrases separated by commas. List each thing once. "
        "Do not list moods, styles, or ideas. Output only the tags.")
PLAIN = "Tag this image. List the things you can see in it. "
WORDINGS = {
    "current": None,
    "no-distinct": "Tag this image. List every visible thing in it: objects, people, animals, clothing, "
                   "materials, and the setting. " + TAIL,
    "no-categories": "Tag this image. List every distinct visible thing in it. " + TAIL,
    "plain": PLAIN + TAIL,
    "plain-stop": PLAIN + TAIL.replace("List each thing once.", "List each thing once, then stop."),
    "plain-prominent": "Tag this image. List the things you can see in it, the most prominent first. " + TAIL,
    "plain-never-twice": PLAIN + TAIL.replace("List each thing once.", "Never write the same tag twice."),
    "plain-up-to-40": "Tag this image. List up to 40 things you can see in it. " + TAIL,
}
# The instruction's own category words, which a reply copies as tags.
CATEGORY_WORDS = {"objects", "animals", "clothing", "materials", "setting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    return parser.parse_args()


def main() -> None:
    ab_env.bootstrap()
    from logit_classifier.tags import complete_tags, repeated_block

    from logit_tagger import engine, tagging

    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    totals = {name: {"loops": 0, "budget": 0, "kept": 0, "rejected": 0, "category": 0, "lost": 0,
                     "gained": 0, "ms": 0.0} for name in WORDINGS}
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
        base_kept: set[str] = set()
        print(f"== {path.name}")
        for name, wording in WORDINGS.items():
            trace = tagging.tag_image(clip, classifier, image, known_things={}, instruction=wording)
            kept = set(trace.tags.split(", ")) if trace.tags else set()
            looped = bool(repeated_block(complete_tags(trace.reply + ",")))
            hit_budget = token_counts[-1] >= tagging.PROPOSE_MAX_TOKENS
            category = sorted(kept & CATEGORY_WORDS)
            total = totals[name]
            if name == "current":
                base_kept = kept
            total["loops"] += looped
            total["budget"] += hit_budget
            total["kept"] += len(kept)
            total["rejected"] += len(trace.candidates) - len(kept)
            total["category"] += len(category)
            total["lost"] += len(base_kept - kept)
            total["gained"] += len(kept - base_kept)
            total["ms"] += trace.propose_ms
            print(f"  {name:17} {token_counts[-1]:3} tokens{' LOOP' if looped else ''}"
                  f"{' BUDGET' if hit_budget else ''}, {len(trace.candidates)} unique, {len(kept)} kept, "
                  f"propose {trace.propose_ms:.0f} ms{', category ' + ' '.join(category) if category else ''}")
            print(f"    raw: {trace.reply!r}")
        print(flush=True)

    print(f"== totals over {len(args.images)} images, gained and lost against current")
    for name, total in totals.items():
        print(f"  {name:17} loops {total['loops']}, budget {total['budget']}, kept {total['kept']}, "
              f"rejected {total['rejected']}, category {total['category']}, gained {total['gained']}, "
              f"lost {total['lost']}, propose {total['ms'] / len(args.images):.0f} ms per image")


if __name__ == "__main__":
    main()
