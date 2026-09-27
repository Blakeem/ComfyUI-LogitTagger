"""Compare ways to end a propose decode that repeats tags, over a real Qwen3-VL-4B CLIP, with no server.

Variants: greedy (the node today), a stop on a repeated block of tags, and core's repetition penalty
through do_sample with top_k 1, which stays deterministic.
Usage: python tests-AB/ab_propose_loops.py <image> [<image> ...]
"""

import argparse
from pathlib import Path

import ab_env
from ab_tagger import DEFAULT_CLIP, load_image

# name: (generate keyword overrides, longest repeated block that stops the decode, 0 for none)
VARIANTS = {
    "greedy": ({}, 0),
    "stop1": ({}, 1),
    "stop4": ({}, 4),
    "pen1.05": ({"repetition_penalty": 1.05}, 0),
    "pen1.1": ({"repetition_penalty": 1.1}, 0),
    "pen1.2": ({"repetition_penalty": 1.2}, 0),
}
SAMPLED_TOP1 = {"do_sample": True, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "temperature": 1.0, "seed": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("images", nargs="+", type=Path, help="image files to tag")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    return parser.parse_args()


def repeated_block(items: list[str], longest: int) -> int:
    """The length of a block of complete items that the list just repeated back to back, else 0."""
    for size in range(1, longest + 1):
        if len(items) >= 2 * size and items[-size:] == items[-2 * size:-size]:
            return size
    return 0


class Decoder:
    """Wraps clip.generate to apply a variant and record how each decode ended."""

    def __init__(self, clip, tagging):
        self.clip = clip
        self.tagging = tagging
        self.generate = clip.generate
        self.transformer = getattr(clip.cond_stage_model, clip.cond_stage_model.clip).transformer
        self.overrides = {}
        self.longest = 0
        self.raw = ""
        self.tokens = 0
        self.ended = ""

    def complete_items(self, ids: list[int]) -> list[str]:
        """Items followed by a separator, so a tag still being written never counts."""
        parts = self.tagging._SEPARATORS.split(self.clip.decode(ids))
        return [self.tagging._clean_item(part) for part in parts[:-1] if self.tagging._clean_item(part)]

    def __call__(self, tokens, **kwargs):
        import torch

        history: list[int] = []
        stopped: list[int] = []
        original = self.transformer.sample_token
        stop_id = self.transformer.model.config.stop_tokens[0]
        ids = None

        def watch(*call_args, **call_kwargs):
            token = original(*call_args, **call_kwargs)
            history.append(int(token.reshape(-1)[0]))
            if self.longest and repeated_block(self.complete_items(history), self.longest):
                stopped.append(len(history))
                return torch.full_like(token, stop_id)
            return token

        kwargs = {**kwargs, **self.overrides}
        self.transformer.sample_token = watch
        try:
            ids = self.generate(tokens, **kwargs)
        finally:
            del self.transformer.sample_token
        self.raw = self.clip.decode(ids)
        self.tokens = len(ids)
        if stopped:
            self.ended = f"loop stop at token {stopped[0]}"
        elif len(ids) >= kwargs["max_length"]:
            self.ended = "hit the budget"
        else:
            self.ended = "stop token"
        return ids


def main() -> None:
    ab_env.bootstrap()
    from logit_tagger import tagging

    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    clip = None
    classifier = None
    decoder = None

    if missing:
        raise SystemExit(f"image files not found: {', '.join(missing)}")

    clip = ab_env.load_clip(args.clip)
    classifier = tagging.build_classifier(clip)
    decoder = Decoder(clip, tagging)
    clip.generate = decoder
    # The first decode pays the model load, so it is run once and discarded.
    tagging.tag_image(clip, load_image(args.images[0]), "", classifier=classifier)

    for path in args.images:
        image = load_image(path)
        base_kept = set()
        print(f"== {path.name}")
        for name, (overrides, longest) in VARIANTS.items():
            decoder.overrides = {**SAMPLED_TOP1, **overrides} if overrides else {}
            decoder.longest = longest
            detail = tagging.tag_image(clip, image, "", classifier=classifier)
            scores = dict(detail.candidates)
            kept = set(detail.tags.split(", ")) if detail.tags else set()
            items = sum(1 for item in tagging._SEPARATORS.split(detail.proposal) if tagging._clean_item(item))
            if name == "greedy":
                base_kept = kept
            print(f"  {name:8} {decoder.tokens:3} tokens, {decoder.ended}, {items} items, "
                  f"{len(detail.candidates)} unique, {len(kept)} kept, propose {detail.propose_ms:.0f} ms")
            if name != "greedy" and kept != base_kept:
                gained = ", ".join(f"{tag} {scores[tag]:.2f}" for tag in sorted(kept - base_kept))
                print(f"           gained: {gained or 'none'}")
                print(f"           lost: {', '.join(sorted(base_kept - kept)) or 'none'}")
            print(f"           raw: {decoder.raw!r}")
        print(flush=True)


if __name__ == "__main__":
    main()
