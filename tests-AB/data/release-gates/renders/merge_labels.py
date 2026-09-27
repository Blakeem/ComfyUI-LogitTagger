"""Merge the two blind judge sets into labels.json, keeping only the answers both judges gave alike.

Judge set A wrote labels_A_*.json and judge set B labels_B_*.json, each following JUDGE.md.
An "unsure" or a disagreement drops that answer. Usage: python merge_labels.py
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEPT = {"checklist": ("yes", "no"), "intended": ("present", "absent")}


def load(judge_set: str) -> dict:
    merged: dict = {}

    for path in sorted(HERE.glob(f"labels_{judge_set}_*.json")):
        merged.update(json.loads(path.read_text(encoding="utf-8")))
    return merged


def merge(left: dict, right: dict) -> tuple[dict, Counter]:
    labels: dict = {}
    stats: Counter = Counter()

    for image_id in sorted(set(left) & set(right)):
        entry: dict = {part: {} for part in KEPT}
        for part, accepted in KEPT.items():
            answers_left = left[image_id].get(part, {})
            answers_right = right[image_id].get(part, {})
            for question, answer in answers_left.items():
                other = answers_right.get(question)
                stats[f"{part} pairs"] += 1
                if answer == other and answer in accepted:
                    entry[part][question] = answer
                    stats[f"{part} agreed"] += 1
                elif answer in accepted and other in accepted:
                    stats[f"{part} conflict"] += 1
        labels[image_id] = entry
    return labels, stats


def main() -> None:
    labels, stats = merge(load("A"), load("B"))

    (HERE / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(len(labels), "images;", dict(stats))


if __name__ == "__main__":
    main()
