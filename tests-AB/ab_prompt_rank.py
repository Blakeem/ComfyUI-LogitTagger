"""Measure whether Qwen3-VL-4B can write image prompt variants from a short request and rank them by a score question.

Decides whether a "prompt brainstorm and rank" node is worth building. Two stages:
  pilot  every instruction wording on the first PILOT_REQUESTS requests, to pick the wording with the fewest failures
  run    one wording on every request, then three text-only score questions per variant and per control
The run stage writes variants_blind.json (requests and variant texts only, for a blind judge) and scores.json.
Usage: python tests-AB/ab_prompt_rank.py pilot|run [--wording NAME] [--graphs on|off] [--out DIR]
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import ab_env
from ab_question_shapes import Asker, auc, mean
from ab_tagger import DEFAULT_CLIP

DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "prompt_rank"

# Each request with a vague one-liner written by hand as its weak control.
REQUESTS = (
    ("a cat on a roof", "a cat on a roof, nice image, high quality"),
    ("cyberpunk street at night", "a street at night, cool style"),
    ("portrait of an old fisherman", "an old man, portrait, good picture"),
    ("a dragon over a castle", "a dragon and a castle, fantasy art"),
    ("cozy reading nook", "a nice room with a chair and books"),
    ("product photo of a perfume bottle", "a perfume bottle, product photo"),
    ("an astronaut in a jungle", "an astronaut and some plants"),
    ("a watercolor of a harbor", "a harbor painting"),
    ("a knight resting after battle", "a knight sitting down"),
    ("a futuristic kitchen", "a modern kitchen, nice"),
    ("a child flying a kite on a beach", "a kid with a kite"),
    ("an abandoned greenhouse", "an old greenhouse, picture"),
)
PILOT_REQUESTS = 3
VARIANTS = 6

# A text starting with <|im_start|> skips core's template, so core adds no empty <think> block.
SYSTEM_TURN = "<|im_start|>system\n{system}<|im_end|>\n"
USER_TURN = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

WORDINGS = {
    "system": (
        "You write prompts for an image generation model. Given a short request, write one detailed prompt "
        "as a single paragraph of 60 to 100 words. Describe the subject, the setting, the composition, the "
        "lighting, and the art style. Output only the prompt, with no title, no list, and no commentary.",
        "{request}",
    ),
    "user": (
        None,
        "Write one detailed image generation prompt for this request: {request}\n"
        "Write a single paragraph of 60 to 100 words that describes the subject, the setting, the composition, "
        "the lighting, and the art style. Output only the prompt.",
    ),
    "split": (
        "You are an expert prompt writer for text-to-image models. You reply with the prompt text only.",
        "Request: {request}\n\nWrite one image prompt as a single paragraph of 60 to 100 words covering the "
        "subject, setting, composition, lighting and style.",
    ),
}

MAX_TOKENS = 200
SAMPLING = {"do_sample": True, "temperature": 0.8, "top_k": 64, "top_p": 0.95, "min_p": 0.05,
            "repetition_penalty": 1.05}
# Qwen3's <|endoftext|> and <|im_end|>, the stop tokens core's Qwen3 configs carry in llama.py.
STOP_TOKENS = frozenset({151643, 151645})

REFUSAL = re.compile(r"\b(I can(?:'|no)t|I cannot|I'm sorry|I am sorry|As an AI|I won't)\b", re.IGNORECASE)
META = re.compile(r"^\W*(here(?:'s| is)|sure|okay|ok\b|certainly|I need to|I will|I'll|let me|prompt\s*:|title\s*:|"
                  r"assistant\b|request\s*:)", re.IGNORECASE)
BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
STOPWORDS = frozenset({"a", "an", "the", "of", "on", "in", "at", "over", "after", "photo", "portrait"})

LEVELS = 5
SCORES = {
    "vivid": ("How vivid and specific is this image prompt",
              ["vague and generic", "somewhat generic", "moderately specific", "vivid and specific",
               "extremely vivid and specific"]),
    "epic": ("How epic and dramatic is the scene this prompt describes",
             ["calm and mundane", "slightly dramatic", "moderately dramatic", "very dramatic",
              "extremely epic and dramatic"]),
    "faithful": ("How faithfully does this prompt keep the request: {request}",
                 ["ignores the request", "loosely related to the request", "partly keeps the request",
                  "mostly keeps the request", "keeps the request exactly"]),
}
CONTROL_KINDS = ("bare", "vague", "swapped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("stage", choices=("pilot", "run"))
    parser.add_argument("--wording", choices=tuple(WORDINGS), default="system", help="run stage wording")
    parser.add_argument("--graphs", choices=("on", "off"), default="on", help="core's CUDA graph decode")
    parser.add_argument("--requests", type=int, default=len(REQUESTS), help="run stage: first N requests")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    return parser.parse_args()


# ---------------------------------------------------------------- generation


def render(wording: str, request: str) -> str:
    system, user = WORDINGS[wording]
    head = SYSTEM_TURN.format(system=system) if system else ""

    return head + USER_TURN.format(user=user.format(request=request))


def clip_generate(clip, tokens, graphs: bool, **kwargs) -> list[int]:
    """engine.clip_generate, with core's CUDA graph decode on or off for the whole process."""
    import comfy.model_management as model_management

    from logit_tagger import engine

    model_management.args.disable_cuda_graphs = not graphs
    return engine.clip_generate(clip, tokens, **kwargs)


def content_words(request: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]+", request.lower()) if word not in STOPWORDS}


def failure(text: str, ids: list[int]) -> str:
    """The first failure the text shows, or "" for none. Ignoring the request is judged by reading."""
    body = text.strip()

    if not body:
        return "empty"
    if REFUSAL.search(body):
        return "refusal"
    if META.search(body) or "<think>" in body:
        return "meta"
    if len(BULLET_LINE.findall(body)) >= 2 or "\n\n" in body:
        return "list"
    if ids and ids[-1] not in STOP_TOKENS:
        return "truncated"
    return ""


def generate_variants(clip, wording: str, requests: list[tuple[str, str]], graphs: bool) -> list[dict]:
    import torch

    rows: list[dict] = []

    for r, (request, _vague) in enumerate(requests, start=1):
        tokens = clip.tokenize(render(wording, request))
        for v in range(1, VARIANTS + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            ids = clip_generate(clip, tokens, graphs, max_length=MAX_TOKENS, seed=1000 * r + v, **SAMPLING)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            text = clip.decode(ids).strip()
            words = content_words(request)
            coverage = len({w for w in words if w in text.lower()}) / len(words)
            rows.append({"id": f"r{r:02d}v{v}", "request": request, "text": text, "tokens": len(ids),
                         "ms": elapsed, "failure": failure(text, ids), "coverage": coverage})
            print(f"{rows[-1]['id']} {len(ids)} tok {elapsed:.0f} ms fail={rows[-1]['failure'] or '-'} "
                  f"cov={coverage:.2f}\n  {text!r}", flush=True)
    return rows


def generation_lines(title: str, rows: list[dict]) -> list[str]:
    failures = [row for row in rows if row["failure"]]
    kinds = sorted({row["failure"] for row in failures})
    tokens = sum(row["tokens"] for row in rows)
    total_ms = sum(row["ms"] for row in rows)

    return [
        f"{title}: {len(failures)}/{len(rows)} failures "
        + (", ".join(f"{kind} {sum(row['failure'] == kind for row in failures)}" for kind in kinds) or ""),
        f"  {tokens / len(rows):.0f} tokens per variant (min {min(r['tokens'] for r in rows)}, "
        f"max {max(r['tokens'] for r in rows)}), {total_ms / len(rows):.0f} ms per variant, "
        f"{total_ms / tokens:.1f} ms per token",
        "  low coverage, read by hand: " + ", ".join(f"{r['id']} {r['coverage']:.2f}" for r in rows
                                                      if r["coverage"] < 0.5),
    ]


# ---------------------------------------------------------------- scoring


def score_questions(request: str) -> dict:
    from logit_classifier import ScoreQuestion

    return {qid: ScoreQuestion(instructions=instructions.format(request=request), criteria=criteria)
            for qid, (instructions, criteria) in SCORES.items()}


def score_text(asker: Asker, request: str, text: str) -> dict:
    answers, ms = asker.ask("score", text, score_questions(request))
    return {"score_ms": ms, **{qid: answers[qid].score for qid in SCORES},
            "probabilities": {qid: answers[qid].probabilities for qid in SCORES}}


def controls(rows: list[dict], requests: list[tuple[str, str]]) -> list[dict]:
    """Per request the bare request, the vague one-liner, and the next request's first variant."""
    items: list[dict] = []

    for r, (request, vague) in enumerate(requests, start=1):
        swapped = rows[(r % len(requests)) * VARIANTS]
        items += [
            {"id": f"r{r:02d}c_bare", "request": request, "kind": "bare", "text": request},
            {"id": f"r{r:02d}c_vague", "request": request, "kind": "vague", "text": vague},
            {"id": f"r{r:02d}c_swapped", "request": request, "kind": "swapped", "text": swapped["text"],
             "source": swapped["id"]},
        ]
    return items


def score_lines(rows: list[dict], control_rows: list[dict]) -> list[str]:
    lines: list[str] = []
    requests = list(dict.fromkeys(row["request"] for row in rows))

    for qid in SCORES:
        per_request = [[row[qid] for row in rows if row["request"] == request] for request in requests]
        spreads = [statistics.pstdev(values) for values in per_request]
        generated = [row[qid] for row in rows]
        weak = [row[qid] for row in control_rows if row["kind"] in ("bare", "vague")]
        lines += [
            f"{qid}: generated mean {mean(generated):.3f}, pooled std {statistics.pstdev(generated):.3f}, "
            f"std of request means {statistics.pstdev([mean(v) for v in per_request]):.3f}",
            f"  within-request std mean {mean(spreads):.3f} (min {min(spreads):.3f}, max {max(spreads):.3f}), "
            f"within-request range mean {mean([max(v) - min(v) for v in per_request]):.3f}",
            f"  AUC generated over bare+vague {auc(generated, weak):.3f}  "
            + "  ".join(f"over {kind} {auc(generated, [c[qid] for c in control_rows if c['kind'] == kind]):.3f} "
                        f"(mean {mean([c[qid] for c in control_rows if c['kind'] == kind]):.3f})"
                        for kind in CONTROL_KINDS),
        ]
    return lines


# ---------------------------------------------------------------- main


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def run_pilot(clip, args: argparse.Namespace) -> None:
    record: dict = {"sampling": SAMPLING, "max_tokens": MAX_TOKENS, "graphs": args.graphs, "wordings": {}}
    lines: list[str] = []

    for wording in WORDINGS:
        rows = generate_variants(clip, wording, list(REQUESTS[:PILOT_REQUESTS]), args.graphs == "on")
        record["wordings"][wording] = rows
        lines += generation_lines(f"wording {wording}", rows)
    print("== pilot", *lines, sep="\n")
    write_json(args.out / "pilot.json", record)


def run_full(clip, args: argparse.Namespace) -> None:
    from logit_classifier import Classifier, Config
    from logit_classifier.backends.comfy_clip import ComfyClipBackend

    requests = list(REQUESTS[:args.requests])
    rows = generate_variants(clip, args.wording, requests, args.graphs == "on")
    control_rows = controls(rows, requests)
    asker = Asker(Classifier(Config(use_prior_debias=False, calibration_path=None), ComfyClipBackend(clip)))
    blind = [{"id": f"r{r:02d}", "request": request,
              "variants": [{"id": row["id"], "text": row["text"]} for row in rows if row["request"] == request]}
             for r, (request, _vague) in enumerate(requests, start=1)]

    # The first call pays CUDA and kernel warmup, which no timing should carry.
    asker.ask("warmup", "warmup", score_questions("warmup"))
    for row in rows + control_rows:
        row.update(score_text(asker, row["request"], row["text"]))
    print("== generation", *generation_lines(f"wording {args.wording}", rows), "== scores",
          *score_lines(rows, control_rows), f"mean {asker.mean_ms('score'):.0f} ms per score request of 3 questions",
          sep="\n")
    write_json(args.out / "variants_blind.json", {"requests": blind})
    write_json(args.out / "scores.json", {
        "clip": args.clip, "wording": args.wording, "instruction": WORDINGS[args.wording], "sampling": SAMPLING,
        "max_tokens": MAX_TOKENS, "graphs": args.graphs, "score_questions": SCORES,
        "variants": {row["id"]: {k: row[k] for k in ("request", "vivid", "epic", "faithful", "probabilities",
                                                     "tokens", "ms", "score_ms", "failure", "coverage")} for row in rows},
        "controls": {row["id"]: {k: v for k, v in row.items() if k != "id"} for row in control_rows},
    })
    print(f"wrote {args.out / 'variants_blind.json'} and {args.out / 'scores.json'}")


def main() -> None:
    ab_env.bootstrap()
    args = parse_args()
    clip = ab_env.load_clip(args.clip)

    # The first generate pays the model load and kernel warmup, which no variant's timing should carry.
    clip_generate(clip, clip.tokenize(render("user", "warmup")), args.graphs == "on", max_length=8, seed=0,
                  **SAMPLING)
    if args.stage == "pilot":
        run_pilot(clip, args)
        return
    run_full(clip, args)


if __name__ == "__main__":
    main()
