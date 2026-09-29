"""Which presence wording separates present from absent tags best, across every grammatical form a tag takes?

The toolkit 0.3.0 wording "Is there {THING} in this image?" adds an article by rule and never says "Are there",
so a plural tag reads "Is there apples in this image?". Each arm writes one yes or no statement per tag, and every
arm asks the same tags of the same image, one packed nouls request per image and arm. ROUND_ARMS holds each round's
arms: round 1 asks whether grammar matters, round 2 searches wordings that need none, round 3 adds the colors
set, where a wording without an article read "orange" as the color, and round 4 ranks the finalists on precision.
The sets, each read at the size its consumer reads:
  coco     build_coco_forms.py, COCO photos with counted forms (singular, plural, irregular, count, mass) and
           related absent things ("cat" beside a dog), 0.5 MP
  renders  data/release-gates, the judged intended things of 72 Krea 2 renders against foreign things, 0.5 MP
  forms    data/presence-forms, 24 Krea 2 renders whose judged tags cover collectives, plural-only nouns, color
           words, possessives, compounds and quoted signs, 0.5 MP
  colors   data/presence-colors, 18 Krea 2 renders asking 12 nouns that also name a color, 0.5 MP
  tiles    every candidate ContextAnchoredTileRefine's shipped tile pass scored (runs/prod-fp8.json), judged yes
           against no or vague by ab_tags_candidates.py, at the 1 MP its verify reads
The precision report ranks by correct tags per image at a fixed pooled precision, so 10 correct and 1 wrong scores
as 20 correct and 2 wrong. Every interval is a 95% bootstrap over images, and each arm's difference to another is
paired image by image.
Usage: python tests-AB/ab_presence_wording.py [--round N] [--sets NAME ...] [--arms NAME ...] [--limit N] [--clip FILE]
              [--out FILE] [--reference ARM] [--detail]
       python tests-AB/ab_presence_wording.py --report FILE ... [--reference ARM] [--detail]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path

import ab_env
from ab_question_shapes import BOOTSTRAP_ROUNDS, SATURATED, auc, bootstrap, interval
from ab_tagger import DEFAULT_CLIP, load_image

HERE = Path(__file__).resolve().parent
COCO_FORMS = HERE / "cache" / "coco-forms" / "coco_forms.json"
RELEASE_GATES = HERE / "data" / "release-gates"
FORMS_DIR = HERE / "data" / "presence-forms"
COLORS_DIR = HERE / "data" / "presence-colors"
TILES_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\custom_nodes\ComfyUI-ContextAnchoredTileRefine\tests-AB"
                 r"\cache\tags_bench")
TILE_RECORD = TILES_DIR / "runs" / "prod-fp8.json"
DEFAULT_OUT = HERE / "out" / "presence-wording" / "results.json"
SETS = ("coco", "renders", "forms", "colors", "tiles")
# The thresholds each consumer reads, in order: the Prompt Check's missing list, the broad verify, the broad tag list
# check, and the strict verify and tag list check.
THRESHOLDS = (0.01, 0.999, 0.9999, 0.99998)
# The pooled precisions the precision report reads correct tags per image at.
PRECISIONS = (0.9, 0.95, 0.97, 0.98, 0.99)
# Scored on its own, since a wrong count is a counting error and not an absent thing.
APART_FORMS = frozenset({"wrong count"})
SEED = 7

# ---------------------------------------------------------------- the is-are arm's grammar

_DETERMINERS = frozenset({"a", "an", "some", "any", "this", "that", "his", "her", "their", "its", "our", "my"})
_PLURAL_DETERMINERS = frozenset({"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                                 "twelve", "dozens", "hundreds", "several", "many", "few", "both", "multiple",
                                 "numerous", "these", "those"})
_HEAD_BREAKS = frozenset({"of", "in", "on", "with", "reading", "under", "over", "at", "from", "holding", "wearing"})
_IRREGULAR_PLURALS = frozenset({
    "people", "children", "men", "women", "mice", "lice", "geese", "teeth", "feet", "dice", "oxen", "cacti", "fungi",
    "knives", "wolves", "leaves", "shelves", "loaves", "halves", "calves", "elves", "wives", "lives", "thieves",
    "scarves", "hooves", "bacteria", "larvae", "antennae", "cattle", "police", "fishermen", "firemen", "policemen",
})
_INVARIANT = frozenset({"sheep", "fish", "deer", "moose", "bison", "salmon", "trout", "shrimp", "squid", "aircraft",
                        "offspring", "series", "species", "swine"})
_PLURAL_ONLY = frozenset({
    "scissors", "glasses", "sunglasses", "eyeglasses", "spectacles", "jeans", "pants", "trousers", "shorts",
    "pajamas", "tongs", "pliers", "binoculars", "goggles", "headphones", "earphones", "clothes", "stairs",
    "tweezers", "leggings", "overalls", "belongings", "surroundings", "remains", "goods", "chips",
})
_S_SINGULARS = frozenset({
    "bus", "gas", "lens", "canvas", "atlas", "iris", "chaos", "news", "physics", "dais", "bias", "corpus",
    "octopus", "cactus", "walrus", "virus", "status", "campus", "circus", "fungus", "abacus", "chorus", "crocus",
    "lotus", "hibiscus", "asparagus", "hummus", "citrus", "platypus", "rhinoceros", "hippopotamus", "bonus",
    "census", "papyrus", "nautilus", "tennis", "texas", "christmas", "mattress", "princess", "compass", "glass",
    "grass", "moss", "dress", "cross", "chess", "harness", "fortress", "witness", "class", "boss",
})
_MASS_NOUNS = frozenset({
    "water", "sand", "snow", "grass", "dirt", "mud", "fog", "mist", "haze", "smoke", "steam", "rain", "ice", "hair",
    "fur", "sky", "sunlight", "moonlight", "starlight", "lightning", "dust", "gravel", "hay", "straw", "rice",
    "bread", "cheese", "butter", "meat", "milk", "coffee", "tea", "beer", "wine", "juice", "soup", "honey", "sugar",
    "salt", "pepper", "flour", "ketchup", "sauce", "oil", "paint", "wood", "timber", "metal", "steel", "iron",
    "brass", "copper", "gold", "silver", "concrete", "marble", "granite", "plastic", "paper", "cardboard", "fabric",
    "cloth", "leather", "silk", "wool", "cotton", "lace", "denim", "velvet", "linen", "jewelry", "furniture",
    "luggage", "clothing", "armor", "armour", "foliage", "moss", "ivy", "seaweed", "lava", "blood", "foam",
    "confetti", "graffiti", "spaghetti", "pasta", "broccoli", "lettuce", "corn", "fruit", "food", "garbage",
    "trash", "rubble", "debris", "machinery", "equipment", "scenery", "vegetation", "traffic", "text", "lettering",
    "writing", "artwork", "signage", "lighting", "darkness", "shade", "fire", "light", "glitter", "rust", "soil",
    "pavement", "asphalt", "chocolate", "cream", "yogurt", "jam", "candy", "dough", "grain", "wheat", "sweat",
    "lavender", "coral", "air", "space", "cash", "money", "ink", "wax", "mail", "sushi", "salad", "toast", "news",
})
# A vowel letter that sounds like a consonant, and an "h" that is silent.
_A_BEFORE = ("uni", "use", "usu", "uti", "ura", "ure", "euro", "eu", "ewe", "one", "once", "ukulele", "utensil")
_AN_BEFORE = ("hour", "honest", "honor", "honour", "heir")


def _words(phrase: str) -> list[str]:
    return [word for word in re.split(r"[\s-]+", phrase.lower().strip()) if word]


def _head(words: list[str]) -> str:
    head_words: list[str] = []

    for word in words:
        if word in _HEAD_BREAKS:
            break
        head_words.append(word)
    return (head_words or words)[-1]


def _plural_word(word: str) -> bool:
    if word in _IRREGULAR_PLURALS or word in _PLURAL_ONLY or word in _INVARIANT:
        return True
    if word in _S_SINGULARS or word.endswith(("ss", "us", "is", "ics")):
        return False
    return word.endswith("s") and len(word) > 2


def _article(first: str) -> str:
    if first.startswith(_A_BEFORE):
        return "a"
    if first.startswith(_AN_BEFORE) or first[0] in "aeiou":
        return "an"
    return "a"


def agreeing_question(tag: str) -> str:
    """ "Is there a dog in this image?", "Are there dogs in this image?", "Is there sand in this image?"."""
    words = _words(tag)
    phrase = tag.strip()

    # "Is there the moon" is not English, so a leading "the" gives way to the article the rest takes.
    if len(words) > 1 and words[0] == "the":
        words, phrase = words[1:], phrase.split(None, 1)[1]
    first = words[0]
    head = _head(words)
    plural = first in _PLURAL_DETERMINERS or first.isdigit() or _plural_word(head)
    if first not in _DETERMINERS and not plural and head not in _MASS_NOUNS:
        phrase = f"{_article(first)} {phrase}"
    return f"{'Are' if plural else 'Is'} there {phrase} in this image?"


def _fill(template: str) -> Callable[[str], str]:
    return lambda tag: template.replace("{TAG}", tag)


# The toolkit 0.3.0 presence_question, copied so the shipped arm stays fixed once the library changes.
_V030_ARTICLES = frozenset({"a", "an", "the", "some"})
_V030_NUMBERS = frozenset({"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"})
_V030_IRREGULAR = frozenset({"people", "children", "cacti", "mice"})
_V030_MASS = frozenset({"water", "butter", "steam", "rain", "sunlight", "dust", "grass", "lightning", "armor", "flour",
                        "fruit", "syrup", "hair", "snow", "cheese", "linen", "dough", "spray"})
_V030_BREAKS = frozenset({"of", "in", "on", "with", "reading"})


def shipped_question(tag: str) -> str:
    """ "Is there {THING} in this image?" with the article toolkit 0.3.0 gave the thing."""
    first = tag.split()[0].lower()
    head_words = [word for word in re.split(r"[\s-]+", tag.lower().strip()) if word]
    head_words = head_words[:next((i for i, w in enumerate(head_words) if w in _V030_BREAKS), len(head_words))]
    head = head_words[-1] if head_words else tag.lower().strip()
    plural = (first in _V030_NUMBERS or first.isdigit() or head in _V030_IRREGULAR or head.endswith("men")
              or (head.endswith("s") and not head.endswith(("ss", "us"))))
    thing = tag
    if not (first in _V030_ARTICLES or first in _V030_NUMBERS or first.isdigit() or plural or head in _V030_MASS):
        thing = f"{'an' if first[0] in 'aeiou' else 'a'} {tag}"
    return f"Is there {thing} in this image?"


def _underscored(template: str) -> Callable[[str], str]:
    return lambda tag: template.replace("{TAG}", "_".join(tag.split()))


# Round 1 asked whether grammar matters. Round 2 searches the wordings that need none.
ROUND_ARMS = {
    1: {
        "is-there": shipped_question,
        "is-are": agreeing_question,
        "visibly-contains": _fill("This image visibly contains {TAG}"),
        "image-contains": _fill("Image contains: {TAG}"),
        "quoted-visible": _fill('Is "{TAG}" visible in this image?'),
        "tag-names": _fill('Does the tag "{TAG}" name something visible in this image?'),
    },
    2: {
        "is-there": shipped_question,
        "visibly-contains": _fill("This image visibly contains {TAG}"),
        "image-contains": _fill("Image contains: {TAG}"),
        "image-contains-quoted": _fill('Image contains "{TAG}"'),
        "this-contains-quoted": _fill('This image contains "{TAG}"'),
        "does-contain-quoted": _fill('Does this image contain "{TAG}"?'),
        "tag-is-visible": _fill("{TAG} is visible in the image"),
        "tag-colon-visible": _fill("{TAG}: visible in this image"),
        "visible-colon": _fill("Visible in this image: {TAG}"),
        "image-shows": _fill("Image shows: {TAG}"),
        "image-depicts": _fill("Image depicts: {TAG}"),
        "in-image": _fill("In this image: {TAG}"),
        "present-colon": _fill("Present in this image: {TAG}"),
        "seen-colon": _fill("Seen in this image: {TAG}"),
        "image-tag": _fill("Image tag: {TAG}"),
        "image-contains-underscored": _underscored("Image contains: {TAG}"),
    },
    # Round 3 adds the colors set, where a label form without an article read "orange" as the color.
    3: {
        "is-there": shipped_question,
        "image-contains": _fill("Image contains: {TAG}"),
        "present-colon": _fill("Present in this image: {TAG}"),
        "visible-colon": _fill("Visible in this image: {TAG}"),
        "does-contain-quoted": _fill('Does this image contain "{TAG}"?'),
        "image-contains-quoted": _fill('Image contains "{TAG}"'),
        "image-depicts": _fill("Image depicts: {TAG}"),
        "present-quoted": _fill('Present in this image: "{TAG}"'),
        "visible-quoted": _fill('Visible in this image: "{TAG}"'),
        "present-thing": _fill("Present in this image, as a thing and not a color: {TAG}"),
    },
    # Round 4 ranks the finalists and the two shipped wordings on precision, with every tile candidate judged.
    4: {
        "is-there": shipped_question,
        "is-are": agreeing_question,
        "visibly-contains": _fill("This image visibly contains {TAG}"),
        "image-contains": _fill("Image contains: {TAG}"),
        "present-colon": _fill("Present in this image: {TAG}"),
        "visible-colon": _fill("Visible in this image: {TAG}"),
        "does-contain-quoted": _fill('Does this image contain "{TAG}"?'),
    },
}


# ---------------------------------------------------------------- sets
# A unit is one image: {"set", "unit", "file", "max_pixels", "items": [{"text", "form", "truth"}]}.
# truth is True or False.


def coco_units(max_pixels: int) -> list[dict]:
    corpus = json.loads(COCO_FORMS.read_text(encoding="utf-8"))

    return [{"set": "coco", "unit": record["file"], "file": COCO_FORMS.parent / record["file"],
             "max_pixels": max_pixels, "items": [{"text": item["text"], "form": item["form"], "truth": item["truth"]}
                                                 for item in record["items"]]}
            for record in corpus["images"]]


def _number(tag: str) -> str:
    words = _words(tag)
    return "plural" if words[0] in _PLURAL_DETERMINERS or _plural_word(_head(words)) else "singular"


def render_units(max_pixels: int) -> list[dict]:
    from ab_release_gates import foreign_things, load_labels, load_renders

    renders = load_renders(RELEASE_GATES / "renders" / "manifest.json")
    labels = load_labels(RELEASE_GATES / "renders" / "labels.json")
    units = []

    for render in renders:
        if render.id not in labels:
            continue
        judged = labels[render.id]["intended"]
        items = [{"text": thing, "form": f"prompt tag {_number(thing)}", "truth": judged[thing] == "present"}
                 for thing in render.intended if thing in judged]
        items += [{"text": thing, "form": f"prompt tag {_number(thing)}", "truth": False}
                  for _source, thing in foreign_things(render, renders, SEED)]
        units.append({"set": "renders", "unit": render.id, "file": render.file, "max_pixels": max_pixels,
                      "items": items})
    return units


def judged_units(set_name: str, folder: Path, max_pixels: int) -> list[dict]:
    """A judged render set: each manifest tag with its form, labeled by the merged judges' labels.json.

    A color word the prompt names only as a color ("orange coat") gets the form "color shown", the hardest absent tag.
    """
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    units = []

    for record in manifest:
        judged = labels.get(record["id"], {})
        forms = record["intended"] | record["distractors"]
        prompt_words = set(re.findall(r"[a-z]+", record["prompt"].lower()))
        items = [{"text": tag, "truth": judged[tag] == "present",
                  "form": "color shown" if tag in record["distractors"] and form == "color-word"
                  and tag in prompt_words else form}
                 for tag, form in forms.items() if tag in judged]
        units.append({"set": set_name, "unit": record["id"], "file": folder / record["file"],
                      "max_pixels": max_pixels, "items": items})
    return units


def tile_units(max_pixels: int) -> list[dict]:
    """Every candidate the tile record scored on a tile, over all its prompts, with the judges' verdict.

    A vague tag counts as wrong, since the tile's text carries it, and an unsure or unsettled one is left out. The form
    is "tile model" when the tile's own list named the tag under any prompt, else "tile prompt".
    """
    record = json.loads(TILE_RECORD.read_text(encoding="utf-8"))
    judgments = json.loads((TILES_DIR / "judgments.json").read_text(encoding="utf-8"))
    truth_of = {"yes": True, "no": False, "vague": False}
    forms: dict[str, dict[str, str]] = {}
    units = []

    for scene_key, by_prompt in record.items():
        for result in by_prompt.values():
            for index, tile in result["tiles"].items():
                seen = forms.setdefault(f"{scene_key}|{int(index):02d}", {})
                for candidate, origin, _score in tile["candidates"]:
                    named_by_model = origin != "prompt" or seen.get(candidate) == "tile model"
                    seen[candidate] = "tile model" if named_by_model else "tile prompt"
    for key, candidates in sorted(forms.items()):
        verdicts = judgments.get(key, {"items": {}})["items"]
        scene, tile = key.split("|")
        items = [{"text": tag, "form": form, "truth": truth_of[verdicts[tag]]}
                 for tag, form in candidates.items() if verdicts.get(tag) in truth_of]
        units.append({"set": "tiles", "unit": key, "file": TILES_DIR / "tiles" / f"{scene}-t{tile}.jpg",
                      "max_pixels": max_pixels, "items": items})
    return units


# ---------------------------------------------------------------- run


def run(units: list[dict], arm_names: list[str], clip_name: str) -> list[dict]:
    from logit_classifier.toolkit.comfyui import fit_image, shared_vision_encode

    from logit_tagger.engine import build_classifier

    arm_of = {name: arm for arms in ROUND_ARMS.values() for name, arm in arms.items()}
    clip = ab_env.load_clip(clip_name)
    classifier = build_classifier(clip, "ab_presence_wording")
    rows: list[dict] = []

    for index, unit in enumerate(units):
        image = fit_image(load_image(unit["file"]), unit["max_pixels"])
        started = time.perf_counter()
        with shared_vision_encode(clip):
            for arm in arm_names:
                statements = [arm_of[arm](item["text"]) for item in unit["items"]]
                scores = classifier.nouls(statements, image=image)
                rows += [{"set": unit["set"], "unit": unit["unit"], "arm": arm, "text": item["text"],
                          "form": item["form"], "truth": item["truth"], "statement": statement, "p": p}
                         for item, statement, p in zip(unit["items"], statements, scores, strict=True)]
        print(f"{index + 1}/{len(units)} {unit['set']} {unit['unit']} {len(unit['items'])} tags "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    return rows


# ---------------------------------------------------------------- report


def _groups(rows: list[dict]) -> list[tuple[list[float], list[float]]]:
    by_unit: dict[str, tuple[list[float], list[float]]] = {}

    for row in rows:
        positives, negatives = by_unit.setdefault(f"{row['set']}|{row['unit']}", ([], []))
        (positives if row["truth"] else negatives).append(row["p"])
    return list(by_unit.values())


def _pooled(groups: list[tuple[list[float], list[float]]], indices: list[int]) -> float:
    return auc([p for i in indices for p in groups[i][0]], [p for i in indices for p in groups[i][1]])


def _scored(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["truth"] is not None and row["form"] not in APART_FORMS]


def _rates(rows: list[dict], threshold: float) -> str:
    positives = [row["p"] for row in rows if row["truth"]]
    negatives = [row["p"] for row in rows if not row["truth"]]
    tpr = sum(p >= threshold for p in positives) / len(positives) if positives else math.nan
    tnr = sum(p < threshold for p in negatives) / len(negatives) if negatives else math.nan
    return f"TPR {tpr:.3f} TNR {tnr:.3f}"


def _auc_line(rows: list[dict]) -> str:
    groups = _groups(rows)
    positives = [p for group in groups for p in group[0]]
    negatives = [p for group in groups for p in group[1]]

    if not positives or not negatives:
        return f"n/a ({len(positives)}+ {len(negatives)}-)"
    spread = interval(bootstrap(lambda indices: _pooled(groups, indices), len(groups), SEED))
    return f"{auc(positives, negatives):.4f} {spread} ({len(positives)}+ {len(negatives)}-)"


def _paired(rows: list[dict], arm: str, reference: str) -> str:
    """AUC(arm) minus AUC(reference) over the same image resamples."""
    import numpy as np

    arm_groups = _groups([row for row in rows if row["arm"] == arm])
    reference_groups = _groups([row for row in rows if row["arm"] == reference])
    difference = (bootstrap(lambda indices: _pooled(arm_groups, indices), len(arm_groups), SEED)
                  - bootstrap(lambda indices: _pooled(reference_groups, indices), len(reference_groups), SEED))
    return f"{float(np.nanmean(difference)):+.4f} {interval(difference)}"


def report(rows: list[dict], arm_names: list[str]) -> list[str]:
    lines: list[str] = []
    sets = [name for name in SETS if any(row["set"] == name for row in rows)]
    scored = _scored(rows)

    for set_name in [*sets, "all"]:
        in_set = [row for row in scored if set_name in ("all", row["set"])]
        forms = sorted({row["form"] for row in in_set})
        lines.append(f"== {set_name}")
        for arm in arm_names:
            arm_rows = [row for row in in_set if row["arm"] == arm]
            lines.append(f"  {arm:<17} AUC {_auc_line(arm_rows)}")
            lines.append("    " + "  ".join(f"@{threshold} {_rates(arm_rows, threshold)}" for threshold in THRESHOLDS))
            lines.append(f"    saturated at p >= {SATURATED}: "
                         f"{sum(row['p'] >= SATURATED for row in arm_rows if row['truth'])} present, "
                         f"{sum(row['p'] >= SATURATED for row in arm_rows if not row['truth'])} absent")
        for reference in ("is-there", "is-are"):
            if reference not in arm_names:
                continue
            lines.append(f"  paired AUC minus {reference}: " + ", ".join(
                f"{arm} {_paired(in_set, arm, reference)}" for arm in arm_names if arm != reference))
        if set_name == "all":
            continue
        lines.append("  per form, AUC, @0.5 rates and paired AUC minus is-there")
        for form in forms:
            form_rows = [row for row in in_set if row["form"] == form]
            lines.append(f"    {form}")
            for arm in arm_names:
                arm_rows = [row for row in form_rows if row["arm"] == arm]
                paired = _paired(form_rows, arm, "is-there") if arm != "is-there" and "is-there" in arm_names else ""
                lines.append(f"      {arm:<17} {_auc_line(arm_rows)}  {_rates(arm_rows, 0.5)}  {paired}")
    lines += apart_lines(rows, arm_names)
    return lines


# The views the summary ranks on. Words leave out counts, since a count tests counting.
VIEWS: dict[str, Callable[[dict], bool]] = {
    "words": lambda row: "count" not in row["form"] and row["set"] != "colors",
    "all": lambda row: True,
    "count": lambda row: row["form"] == "count",
    "related": lambda row: row["set"] == "coco" and (row["form"] == "related"
                                                    or (row["form"] == "singular" and row["truth"])),
    "tiles": lambda row: row["set"] == "tiles",
    "colors": lambda row: row["set"] == "colors",
}


def _point_auc(rows: list[dict]) -> float:
    return auc([row["p"] for row in rows if row["truth"]], [row["p"] for row in rows if not row["truth"]])


def summary(rows: list[dict], arm_names: list[str], reference: str) -> list[str]:
    """One line per arm, ranked by AUC on words: every view's AUC, the paired words difference to the reference,
    and the tile tags each keeps at the tagger's 0.999."""
    scored = _scored(rows)
    view_rows = {view: [row for row in scored if keep(row)] for view, keep in VIEWS.items()}
    ranked = sorted(arm_names, key=lambda arm: -_point_auc([r for r in view_rows["words"] if r["arm"] == arm]))
    multiword = [row for row in scored if " " in row["text"].strip()]
    lines = [f"== summary, ranked by AUC on words; paired words AUC minus {reference}",
             f"  {'arm':<27} {'words':>7} {'all':>7} {'count':>7} {'related':>7} {'tiles':>7} {'colors':>7}  "
             f"{'tiles@0.999 TPR/TNR':>19}  paired words"]

    for arm in ranked:
        aucs = [_point_auc([row for row in view_rows[view] if row["arm"] == arm]) for view in VIEWS]
        tile_rows = [row for row in view_rows["tiles"] if row["arm"] == arm]
        paired = _paired(view_rows["words"], arm, reference) if arm != reference and reference in arm_names else ""
        lines.append(f"  {arm:<27} " + " ".join(f"{value:7.4f}" for value in aucs)
                     + f"  {_rates(tile_rows, 0.999):>19}  {paired}")
    lines.append("  words AUC with interval: " + ", ".join(
        f"{arm} {_auc_line([row for row in view_rows['words'] if row['arm'] == arm])}" for arm in ranked[:4]))
    if "image-contains-underscored" in arm_names and reference in arm_names:
        lines.append(f"  underscores on {len({(r['set'], r['unit'], r['text']) for r in multiword})} multi-word tags, "
                     f"paired AUC minus {reference}: {_paired(multiword, 'image-contains-underscored', reference)}")
    shown = [row for row in scored if row["form"] == "color shown"]
    if shown:
        lines.append("  color shown, thing absent, read present at p >= 0.5: " + ", ".join(
            f"{arm} {sum(r['p'] >= 0.5 for r in shown if r['arm'] == arm)}/{sum(r['arm'] == arm for r in shown)}"
            for arm in ranked))
    return lines


# The views the precision report reads, one per consumer's kind of tag. A tile tag is a candidate the model proposed,
# a coco or forms tag is a declared tag, and a render tag is a prompt's thing.
PRECISION_VIEWS: dict[str, Callable[[dict], bool]] = {
    "tiles": lambda row: row["set"] == "tiles",
    "tiles, model's own tags": lambda row: row["form"] == "tile model",
    "tiles, prompt tags": lambda row: row["form"] == "tile prompt",
    "coco words": lambda row: row["set"] == "coco" and "count" not in row["form"],
    "coco counts": lambda row: row["set"] == "coco" and row["form"] == "count",
    "renders": lambda row: row["set"] == "renders",
    "forms": lambda row: row["set"] == "forms",
    "colors": lambda row: row["set"] == "colors",
}


def _ordered(rows: list[dict], unit_index: dict[str, int]):
    """The rows' truths and images sorted by score, high first, with a mask on the last row of each run of tied scores."""
    import numpy as np

    order = sorted(rows, key=lambda row: -row["p"])
    scores = np.array([row["p"] for row in order])
    truths = np.array([bool(row["truth"]) for row in order])
    units = np.array([unit_index[f"{row['set']}|{row['unit']}"] for row in order])
    last_of_tie = np.append(scores[1:] != scores[:-1], True)
    return truths, units, last_of_tie


def _kept_at(ordered, weights, target: float) -> tuple[float, float]:
    """Correct and wrong tags per image, and the threshold, at the threshold keeping the most correct tags while the
    pooled precision stays at target or above. Each image counts `weights[image]` times, so a bootstrap resample is a
    weight vector."""
    import numpy as np

    truths, units, last_of_tie = ordered
    row_weights = weights[units]
    correct = np.cumsum(row_weights * truths)[last_of_tie]
    wrong = np.cumsum(row_weights * ~truths)[last_of_tie]
    precision = correct / np.maximum(correct + wrong, 1e-12)
    reaching = np.flatnonzero(precision >= target)
    images = weights.sum()

    if reaching.size == 0:
        return 0.0, 0.0
    best = reaching[np.argmax(correct[reaching])]
    return correct[best] / images, wrong[best] / images


def _at_threshold(rows: list[dict], threshold: float, images: int) -> str:
    kept = [row for row in rows if row["p"] >= threshold]
    correct = sum(bool(row["truth"]) for row in kept)
    precision = correct / len(kept) if kept else math.nan
    return f"@{threshold} {precision:.3f} {correct / images:5.2f} {(len(kept) - correct) / images:4.2f}"


def precision_lines(rows: list[dict], arm_names: list[str], reference: str) -> list[str]:
    """Per view, the correct tags per image each arm keeps at every pooled precision in PRECISIONS, the paired
    difference to the reference there, and the precision and tags per image at the thresholds the consumers read."""
    import numpy as np

    scored = _scored(rows)
    lines = ["== precision: correct tags per image at pooled precision P (wrong per image), and at fixed thresholds "
             "precision, correct and wrong per image"]

    for view, keep in PRECISION_VIEWS.items():
        in_view = [row for row in scored if keep(row)]
        if not in_view:
            continue
        unit_names = sorted({f"{row['set']}|{row['unit']}" for row in in_view})
        unit_index = {name: i for i, name in enumerate(unit_names)}
        ordered = {arm: _ordered([row for row in in_view if row["arm"] == arm], unit_index) for arm in arm_names}
        ones = np.ones(len(unit_names))
        reference_rows = [row for row in in_view if row["arm"] == (reference if reference in arm_names else arm_names[0])]
        present = sum(bool(row["truth"]) for row in reference_rows)
        lines.append(f"-- {view}: {len(unit_names)} images, {present} present and {len(reference_rows) - present} "
                     f"absent tags, base precision {present / len(reference_rows):.3f}")
        lines.append(f"  {'arm':<20}" + "".join(f"{'P' + str(target):>13}" for target in PRECISIONS)
                     + "   fixed thresholds")
        for arm in arm_names:
            cells = [_kept_at(ordered[arm], ones, target) for target in PRECISIONS]
            fixed = [_at_threshold([row for row in in_view if row["arm"] == arm], threshold, len(unit_names))
                     for threshold in (0.999, 0.9999)]
            lines.append(f"  {arm:<20}" + "".join(f"{correct:7.2f} ({wrong:.2f})" for correct, wrong in cells)
                         + "   " + "  ".join(fixed))
        if reference not in arm_names:
            continue
        rng = np.random.default_rng(SEED)
        resamples = [np.bincount(rng.integers(0, len(unit_names), len(unit_names)), minlength=len(unit_names))
                     for _ in range(BOOTSTRAP_ROUNDS)]
        for target in (0.95, 0.98):
            reference_kept = np.array([_kept_at(ordered[reference], weights, target)[0] for weights in resamples])
            differences = []
            for arm in arm_names:
                if arm == reference:
                    continue
                arm_kept = np.array([_kept_at(ordered[arm], weights, target)[0] for weights in resamples])
                point = _kept_at(ordered[arm], ones, target)[0] - _kept_at(ordered[reference], ones, target)[0]
                differences.append(f"{arm} {point:+.2f} {interval(arm_kept - reference_kept)}")
            lines.append(f"  paired at P{target} minus {reference}: " + ", ".join(differences))
    return lines


def apart_lines(rows: list[dict], arm_names: list[str]) -> list[str]:
    """The tags outside the pooled rows: a wrong count should read no."""
    lines = ["== scored apart"]

    for form in APART_FORMS:
        for arm in arm_names:
            chosen = [row["p"] for row in rows if row["form"] == form and row["arm"] == arm]
            if chosen:
                lines.append(f"  {form:<11} {arm:<17} n {len(chosen)}  "
                             + "  ".join(f"p >= {t} {sum(p >= t for p in chosen) / len(chosen):.3f}" for t in THRESHOLDS))
    return lines


def disagreements(rows: list[dict], first: str, second: str, count: int = 15) -> list[str]:
    """The tags where two arms differ most, for reading which wording errs."""
    by_key = {(row["set"], row["unit"], row["text"]): row for row in rows if row["arm"] == first}
    pairs = [(by_key[(row["set"], row["unit"], row["text"])], row) for row in rows
             if row["arm"] == second and row["truth"] is not None and (row["set"], row["unit"], row["text"]) in by_key]
    pairs.sort(key=lambda pair: abs(pair[0]["p"] - pair[1]["p"]), reverse=True)
    return [f"== largest {first} vs {second} gaps"] + [
        f"  {a['set']} {a['unit']} {a['text']!r} truth {a['truth']}  {first} {a['p']:.4f}  {second} {b['p']:.4f}"
        for a, b in pairs[:count]]


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS), help="sets to run")
    parser.add_argument("--round", type=int, choices=sorted(ROUND_ARMS), default=4, help="the round's arms, default 4")
    parser.add_argument("--arms", nargs="+", default=None, help="arms to run, default every arm of the round")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images of each set")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"raw rows JSON, default {DEFAULT_OUT}")
    parser.add_argument("--report", type=Path, nargs="+", default=None,
                        help="print the report of one or more results files, their rows joined, and exit")
    parser.add_argument("--detail", action="store_true", help="add every set's per-arm and per-form rows")
    parser.add_argument("--reference", default="visibly-contains",
                        help="the arm the precision report pairs every other arm with, default visibly-contains")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    rows: list[dict] = []
    units: list[dict] = []

    if args.report is not None:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
        rows = [row for record in records for row in record["rows"]]
        arm_names = records[0]["arms"]
        print(*precision_lines(rows, arm_names, args.reference), *summary(rows, arm_names, args.reference),
              *(report(rows, arm_names) if args.detail else ()),
              *disagreements(rows, "is-there", args.reference), sep="\n")
        return
    ab_env.bootstrap()
    from logit_tagger.engine import CHECK_MAX_PIXELS, VL_MAX_PIXELS

    arm_names = args.arms or list(ROUND_ARMS[args.round])
    loaders = {"coco": lambda: coco_units(CHECK_MAX_PIXELS), "renders": lambda: render_units(CHECK_MAX_PIXELS),
               "forms": lambda: judged_units("forms", FORMS_DIR, CHECK_MAX_PIXELS),
               "colors": lambda: judged_units("colors", COLORS_DIR, CHECK_MAX_PIXELS),
               "tiles": lambda: tile_units(VL_MAX_PIXELS)}
    # Every input is read before the CLIP loads, so a missing file fails in seconds.
    for name in args.sets:
        units += loaders[name]()[: args.limit]
    missing = [str(unit["file"]) for unit in units if not Path(unit["file"]).is_file()]
    if missing:
        raise SystemExit(f"images not found: {', '.join(missing)}")
    rows = run(units, arm_names, args.clip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clip": args.clip, "arms": arm_names, "sets": args.sets, "limit": args.limit,
                                    "rows": rows}, indent=1), encoding="utf-8")
    print(*precision_lines(rows, arm_names, args.reference), *summary(rows, arm_names, args.reference),
          *(report(rows, arm_names) if args.detail else ()), sep="\n")
    print(f"raw rows: {args.out}")


if __name__ == "__main__":
    main()
