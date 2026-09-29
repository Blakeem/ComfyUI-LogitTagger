"""Render the presence forms set, Krea 2 renders whose tags cover the grammatical forms COCO lacks, through a running ComfyUI server.

Each scene lists its intended tags and its distractor tags, both with their form. The distractors are things the
prompt never names, of the same forms, so the absent side of every form has examples. Judges label every tag of
both lists (JUDGE.md). The graph, models and settings are those of release-gates/renders/render.py. Re-running
skips ids already rendered.

    python render_forms.py [--only f01,f02] [--limit N]
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED_BASE = 3000

_spec = importlib.util.spec_from_file_location("render", HERE.parent / "release-gates" / "renders" / "render.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

FORMS = frozenset({"singular", "plural", "irregular", "count", "mass", "plural-only", "collective", "color-word",
                   "vowel-sound", "s-singular", "possessive", "quoted", "compound"})

# (style, size, prompt, {intended tag: form}, {distractor tag: form})
SCENES = [
    ("photograph", render.LANDSCAPE,
     "A shepherd guiding a flock of sheep across a green hillside at dawn, two dogs running beside them, low fog "
     "in the valley below, old stone walls, overcast sky.",
     {"shepherd": "singular", "flock of sheep": "collective", "sheep": "irregular", "two dogs": "count",
      "fog": "mass", "stone walls": "plural", "hillside": "singular"},
     {"geese": "irregular", "cows": "plural", "tractor": "singular", "snow": "mass", "three horses": "count"}),
    ("photograph", render.WIDE,
     "A farmyard with a flock of white geese walking past a red wooden barn, a pile of hay bales, chickens "
     "pecking at the mud, a rusty wheelbarrow, soft morning light.",
     {"geese": "irregular", "red barn": "singular", "hay bales": "plural", "hay": "mass", "chickens": "plural",
      "mud": "mass", "wheelbarrow": "singular"},
     {"sheep": "irregular", "pigs": "plural", "tractor": "singular", "water": "mass", "mice": "irregular"}),
    ("photograph", render.PORTRAIT,
     "Close-up photograph of a smiling woman with bright white teeth, silver hoop earrings and a pair of glasses, "
     "holding a slice of orange near her face, studio lighting, pastel background.",
     {"woman": "singular", "teeth": "irregular", "earrings": "plural", "glasses": "plural-only",
      "slice of orange": "collective", "orange": "color-word", "face": "singular"},
     {"necklace": "singular", "lime": "color-word", "hat": "singular", "scissors": "plural-only",
      "children": "irregular"}),
    ("photograph", render.SQUARE,
     "Overhead photograph of a kitchen counter with a bowl of rice, a loaf of bread, three eggs, a pair of "
     "scissors, a chef's knife on a cutting board, and a glass of milk, natural window light.",
     {"bowl of rice": "collective", "rice": "mass", "loaf of bread": "collective", "bread": "mass",
      "three eggs": "count", "scissors": "plural-only", "chef's knife": "possessive", "cutting board": "singular",
      "glass of milk": "collective"},
     {"spoons": "plural", "lemon": "singular", "sugar": "mass", "two cups": "count", "honey": "mass"}),
    ("photograph", render.LANDSCAPE,
     "Photograph of a street fruit stall piled with oranges, limes, peaches and plums in wooden crates, a vendor "
     "in a straw hat, a hand-written sign reading \"FRESH\", afternoon sun.",
     {"oranges": "color-word", "limes": "color-word", "peaches": "color-word", "plums": "color-word",
      "wooden crates": "plural", "vendor": "singular", "straw hat": "singular", "sign reading FRESH": "quoted"},
     {"bananas": "plural", "watermelons": "plural", "cash register": "singular", "cherries": "color-word",
      "sign reading CLOSED": "quoted"}),
    ("photograph", render.WIDE,
     "Aquarium photograph of a school of small silver fish swimming around an orange octopus on a coral reef, "
     "sea anemones, bubbles rising through the clear blue water.",
     {"fish": "irregular", "school of fish": "collective", "octopus": "s-singular", "coral": "color-word",
      "sea anemones": "plural", "bubbles": "plural", "water": "mass"},
     {"shark": "singular", "turtles": "plural", "diver": "singular", "whale": "singular", "seaweed": "mass"}),
    ("photograph", render.LANDSCAPE,
     "Photograph of a desert landscape with a tall saguaro cactus, rolling sand dunes, a lizard resting on a red "
     "rock, a clear blue sky, harsh midday sun.",
     {"cactus": "s-singular", "sand dunes": "plural", "sand": "mass", "lizard": "singular", "rock": "singular",
      "blue sky": "singular"},
     {"camels": "plural", "snow": "mass", "snake": "singular", "palm trees": "plural", "water": "mass"}),
    ("photograph", render.WIDE,
     "Two children in red coats building a snowman in a snowy park, mittens on their hands, a wooden sled leaning "
     "against a bench, bare trees, snow falling.",
     {"children": "irregular", "two children": "count", "red coats": "plural", "snowman": "singular",
      "snow": "mass", "mittens": "plural", "sled": "singular", "bench": "singular"},
     {"dog": "singular", "ice skates": "plural", "men": "irregular", "puddles": "plural", "grass": "mass"}),
    ("photograph", render.LANDSCAPE,
     "Autumn forest path covered in fallen leaves, a deer and two fawns standing among birch trees, mist between "
     "the trunks, a wooden signpost, soft golden light.",
     {"leaves": "irregular", "deer": "irregular", "birch trees": "plural", "mist": "mass",
      "wooden signpost": "singular", "path": "singular"},
     {"wolves": "irregular", "bear": "singular", "snow": "mass", "river": "singular", "mushrooms": "plural"}),
    ("photograph", render.LANDSCAPE,
     "A cozy library with tall wooden shelves full of books, a stack of books on a leather armchair, an hourglass "
     "and a brass compass on the desk, a green lamp, warm evening light.",
     {"shelves": "irregular", "books": "plural", "stack of books": "collective", "leather armchair": "singular",
      "hourglass": "vowel-sound", "brass compass": "s-singular", "desk": "singular", "green lamp": "singular"},
     {"globe": "singular", "typewriter": "singular", "candles": "plural", "cat": "singular",
      "umbrella": "vowel-sound"}),
    ("photograph", render.WIDE,
     "A woman with an umbrella crossing a rainy city street at night, a yellow bus passing, puddles reflecting "
     "neon lights, a cyclist behind her, a shop sign reading \"OPEN\".",
     {"woman": "singular", "umbrella": "vowel-sound", "bus": "s-singular", "puddles": "plural",
      "neon lights": "plural", "cyclist": "singular", "sign reading OPEN": "quoted", "rain": "mass"},
     {"taxis": "plural", "dog": "singular", "snow": "mass", "tram": "singular", "children": "irregular"}),
    ("fantasy illustration", render.LANDSCAPE,
     "Fantasy illustration of a white unicorn standing in a glowing forest clearing, a ring of red mushrooms, "
     "fireflies drifting in the air, a stone archway covered in moss, a crescent moon in the night sky.",
     {"unicorn": "vowel-sound", "red mushrooms": "plural", "fireflies": "plural", "stone archway": "singular",
      "moss": "mass", "crescent moon": "singular", "night sky": "singular"},
     {"dragon": "singular", "castle": "singular", "fairies": "plural", "river": "singular", "owls": "plural"}),
    ("anime illustration", render.PORTRAIT,
     "Anime illustration of a girl in a school uniform feeding three white mice from her palm, sunflowers in a "
     "window box behind her, warm sunlight, pastel colors, soft linework.",
     {"girl": "singular", "school uniform": "vowel-sound", "mice": "irregular", "three mice": "count",
      "sunflowers": "plural", "window box": "singular", "sunlight": "mass"},
     {"cat": "singular", "birds": "plural", "bicycle": "singular", "cherry blossoms": "plural",
      "teacup": "singular"}),
    ("oil painting", render.LANDSCAPE,
     "Still life oil painting of a copper kettle, a bunch of grapes, two lemons, a sprig of lavender in a glass "
     "vase, a wheel of cheese and scattered walnuts on a linen cloth.",
     {"copper kettle": "singular", "bunch of grapes": "collective", "grapes": "plural", "two lemons": "count",
      "lavender": "color-word", "glass vase": "s-singular", "wheel of cheese": "collective", "cheese": "mass",
      "walnuts": "plural"},
     {"pear": "singular", "bread": "mass", "candle": "singular", "three apples": "count", "wine": "mass"}),
    ("oil painting", render.WIDE,
     "Oil painting of fishermen hauling nets onto a wooden boat at sunset, seagulls circling above, waves crashing "
     "on the rocks, a lighthouse on the cliff, warm amber light.",
     {"fishermen": "irregular", "nets": "plural", "wooden boat": "singular", "seagulls": "plural",
      "waves": "plural", "rocks": "plural", "lighthouse": "singular", "cliff": "singular"},
     {"whale": "singular", "children": "irregular", "bridge": "singular", "snow": "mass", "horses": "plural"}),
    ("3D render", render.SQUARE,
     "3D render of a toy robot holding a red balloon on a wooden table, a pair of headphones, scattered dice, a "
     "potted succulent and a coffee mug, soft studio lighting, clean background.",
     {"toy robot": "singular", "red balloon": "singular", "wooden table": "singular",
      "pair of headphones": "collective", "headphones": "plural-only", "dice": "irregular",
      "potted succulent": "singular", "coffee mug": "singular"},
     {"teddy bear": "singular", "glasses": "plural-only", "pencils": "plural", "laptop": "singular",
      "keys": "plural"}),
    ("photograph", render.WIDE,
     "Aerial photograph of a busy harbor with container ships, cranes loading cargo, stacks of colorful shipping "
     "containers, small tugboats, and a long bridge in the distance, hazy morning.",
     {"container ships": "plural", "cranes": "plural", "shipping containers": "plural",
      "stacks of containers": "collective", "tugboats": "plural", "bridge": "singular"},
     {"airplanes": "plural", "lighthouse": "singular", "sailboats": "plural", "beach": "singular",
      "people": "irregular"}),
    ("photograph", render.PORTRAIT,
     "Portrait photograph of a bearded man with blue eyes wearing a wool beanie and a denim jacket, holding a cup "
     "of coffee with steam rising, a red brick wall behind him.",
     {"man": "singular", "beard": "singular", "blue eyes": "plural", "wool beanie": "singular",
      "denim jacket": "singular", "cup of coffee": "collective", "steam": "mass", "brick wall": "singular"},
     {"glasses": "plural-only", "scarf": "singular", "woman": "singular", "headphones": "plural-only",
      "tie": "singular"}),
    ("photograph", render.WIDE,
     "Beach photograph of people playing volleyball on the sand, a row of colorful umbrellas, surfboards standing "
     "in the sand, palm trees and turquoise waves, bright summer day.",
     {"people": "irregular", "volleyball": "singular", "sand": "mass", "row of umbrellas": "collective",
      "umbrellas": "plural", "surfboards": "plural", "palm trees": "plural", "waves": "plural"},
     {"horse": "singular", "snow": "mass", "kites": "plural", "car": "singular", "geese": "irregular"}),
    ("pixel art", render.LANDSCAPE,
     "Pixel art of a knight in armor fighting a dragon near a castle, a treasure chest full of gold coins, flames, "
     "torches on the stone walls, dark dungeon colors, retro game style.",
     {"knight": "singular", "armor": "mass", "dragon": "singular", "castle": "singular",
      "treasure chest": "singular", "gold coins": "color-word", "flames": "plural", "torches": "plural"},
     {"wizard": "singular", "horses": "plural", "bow and arrow": "compound", "skeletons": "plural",
      "princess": "s-singular"}),
    ("watercolor", render.PORTRAIT,
     "Watercolor painting of a bouquet of roses and tulips in a blue jug, a pair of reading glasses and an open "
     "letter on the table beside it, soft washes, white paper background.",
     {"bouquet of roses": "collective", "roses": "color-word", "tulips": "plural", "blue jug": "singular",
      "reading glasses": "plural-only", "open letter": "vowel-sound", "table": "singular"},
     {"sunflowers": "plural", "teapot": "singular", "pen": "singular", "butterflies": "plural",
      "candle": "singular"}),
    ("comic book", render.PORTRAIT,
     "Comic book panel of a superhero in a red cape flying over the city skyline, a helicopter with a searchlight, "
     "a speech bubble reading \"UP, UP!\", skyscrapers and clouds, bold ink outlines.",
     {"superhero": "singular", "red cape": "singular", "city skyline": "singular", "helicopter": "singular",
      "searchlight": "singular", "speech bubble": "singular", "skyscrapers": "plural", "clouds": "plural"},
     {"train": "singular", "dragon": "singular", "hot air balloon": "singular", "fireworks": "plural",
      "people": "irregular"}),
    ("photograph", render.WIDE,
     "Backyard photograph of a barbecue grill with sausages and corn on the cob, smoke rising, a family of four "
     "sitting at a picnic table, a dog waiting nearby, green grass and a wooden fence.",
     {"barbecue grill": "singular", "sausages": "plural", "corn on the cob": "collective", "smoke": "mass",
      "four people": "count", "picnic table": "singular", "dog": "singular", "grass": "mass",
      "wooden fence": "singular"},
     {"swimming pool": "singular", "cats": "plural", "fish": "irregular", "wine glasses": "plural",
      "trampoline": "singular"}),
    ("photograph", render.SQUARE,
     "Pub photograph of fish and chips on a white plate with a lemon wedge, salt and pepper shakers, a pint of "
     "beer, a small dish of ketchup, raindrops on the window behind.",
     {"fish and chips": "compound", "white plate": "singular", "lemon wedge": "singular",
      "salt and pepper shakers": "compound", "pint of beer": "collective", "beer": "mass", "ketchup": "mass",
      "raindrops": "plural", "window": "singular"},
     {"burger": "singular", "candles": "plural", "knife and fork": "compound", "newspaper": "singular",
      "soup": "mass"}),
]


def validate_scenes(scenes: list, id_prefix: str) -> None:
    for index, (_style, _size, prompt, intended, distractors) in enumerate(scenes, start=1):
        words = len(prompt.split())
        overlap = set(intended) & set(distractors)
        forms = set(intended.values()) | set(distractors.values())
        if not 20 <= words <= 60:
            raise ValueError(f"{id_prefix}{index:02d} has {words} words, outside 20-60")
        if overlap:
            raise ValueError(f"{id_prefix}{index:02d} lists {sorted(overlap)} as both intended and distractor")
        if forms - FORMS:
            raise ValueError(f"{id_prefix}{index:02d} uses unknown forms {sorted(forms - FORMS)}")


def render_set(scenes: list, out_dir: Path, id_prefix: str, seed_base: int, argv: list[str] | None = None) -> int:
    """Render every scene not yet in out_dir's manifest.json, and record each one there as it lands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help=f"comma separated ids, e.g. {id_prefix}01,{id_prefix}02")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new renders")
    args = parser.parse_args(argv)
    manifest = out_dir / "manifest.json"
    only_ids = {item.strip() for item in args.only.split(",") if item.strip()}
    records = ({record["id"]: record for record in json.loads(manifest.read_text(encoding="utf-8"))}
               if manifest.is_file() else {})
    rendered = 0
    failures = []

    validate_scenes(scenes, id_prefix)
    for index, (style, (width, height), prompt, intended, distractors) in enumerate(scenes, start=1):
        image_id = f"{id_prefix}{index:02d}"
        if (only_ids and image_id not in only_ids) or (args.limit and rendered >= args.limit):
            continue
        if image_id in records and (out_dir / records[image_id]["file"]).is_file() and not only_ids:
            continue
        seed = seed_base + index
        graph = render.build_graph(prompt, width, height, seed, render.STEPS, f"LT-{out_dir.name}/{image_id}")
        try:
            source, seconds = render.run_prompt(graph)
        except (RuntimeError, OSError) as error:
            failures.append(f"{image_id}: {error}")
            print(f"{image_id} FAILED {error}", flush=True)
            continue
        shutil.copy2(source, out_dir / f"{image_id}.png")
        records[image_id] = {"id": image_id, "file": f"{image_id}.png", "prompt": prompt, "style": style,
                             "intended": intended, "distractors": distractors, "width": width, "height": height,
                             "seed": seed, "steps": render.STEPS, "seconds": round(seconds, 2)}
        manifest.write_text(json.dumps([records[key] for key in sorted(records)], indent=2, ensure_ascii=False),
                            encoding="utf-8")
        rendered += 1
        print(f"{image_id} {style} {width}x{height} {seconds:.1f}s", flush=True)

    print(f"rendered {rendered}, failed {len(failures)}, manifest has {len(records)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(render_set(SCENES, HERE, "f", SEED_BASE))
