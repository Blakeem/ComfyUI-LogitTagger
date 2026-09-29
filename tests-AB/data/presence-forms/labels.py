"""Split a judged render set (forms, or presence-colors with --dir) into blind judge batches, then merge two judge sets into labels.json.

split writes judge_in_<set>_<n>.json, each mapping image id to {"file", "tags"} with the intended and distractor
tags shuffled together, so a judge cannot tell them apart. Judges write labels_<set>_<n>.json following JUDGE.md.
merge keeps an answer only when both judge sets gave the same "present" or "absent".
Usage: python labels.py split [--dir DIR] [--batches N]
       python labels.py merge [--dir DIR]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE_SETS = ("A", "B")
KEPT = ("present", "absent")
SEED = 5


def split(folder: Path, batches: int) -> None:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

    for judge_set in JUDGE_SETS:
        rng = random.Random(f"{SEED}-{judge_set}")
        for batch in range(batches):
            chosen = manifest[batch::batches]
            entries = {}
            for record in chosen:
                tags = [*record["intended"], *record["distractors"]]
                rng.shuffle(tags)
                entries[record["id"]] = {"file": str(folder / record["file"]), "tags": tags}
            path = folder / f"judge_in_{judge_set}_{batch + 1}.json"
            path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
            print(f"{path.name}: {len(entries)} images")


def load(folder: Path, judge_set: str) -> dict:
    merged: dict = {}

    for path in sorted(folder.glob(f"labels_{judge_set}_*.json")):
        merged.update(json.loads(path.read_text(encoding="utf-8")))
    return merged


def merge(folder: Path) -> None:
    left, right = (load(folder, judge_set) for judge_set in JUDGE_SETS)
    labels: dict = {}
    stats: Counter = Counter()

    for image_id in sorted(set(left) & set(right)):
        labels[image_id] = {}
        for tag, answer in left[image_id].items():
            other = right[image_id].get(tag)
            stats["pairs"] += 1
            if answer == other and answer in KEPT:
                labels[image_id][tag] = answer
                stats[f"agreed {answer}"] += 1
            elif answer in KEPT and other in KEPT:
                stats["conflict"] += 1
            else:
                stats["unsure or missing"] += 1
    (folder / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"{len(labels)} images; {dict(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("split", "merge"))
    parser.add_argument("--batches", type=int, default=2, help="batches per judge set, default 2")
    parser.add_argument("--dir", type=Path, default=HERE, help="the set's folder, holding manifest.json, default this one")
    args = parser.parse_args()

    if args.command == "split":
        split(args.dir.resolve(), args.batches)
    else:
        merge(args.dir.resolve())


if __name__ == "__main__":
    main()
