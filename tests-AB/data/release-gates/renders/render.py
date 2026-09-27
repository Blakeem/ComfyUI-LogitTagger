"""Render the Logit Tagger synthetic test set through a running ComfyUI server.

The graph is the base text-to-image path of the owner's `Krea 2.json` workflow. The bypassed
control LoRA and the tile upscale branch are left out. Re-running skips ids already rendered.

    python render.py [--steps N] [--only g01,g02] [--limit N]
"""

import argparse
import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

SERVER = "http://127.0.0.1:8199"
OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output")
RENDERS_DIR = Path(__file__).resolve().parent
MANIFEST = RENDERS_DIR / "manifest.json"

UNET = "krea2Center_v408bitFp8Int8.safetensors"
CLIP = "qwen3-vl-4b-heretic_int8.safetensors"
VAE = "krea2RealVae_v10.safetensors"
SAMPLER = "dpmpp_2m"
SCHEDULER = "sgm_uniform"
# The workflow uses 52 steps. 26 keeps a render under 40 s on the 3090 Ti (52 took 75 s).
STEPS = 26
CFG = 3.5
SEED_BASE = 1000
NEGATIVE = (
    "blurry, hazy, smoke, fog, grandparent, geriatric, fat, overweight, obese, muscular, "
    "bodybuilder, fitness body, ample bosom, big breasts, busty, voluptuous hourglass figure, "
    "body piercings, navel piercing, body hair, disproportionate body, pregnant, tan line, "
    "jpeg compression, scribbles, AI generated, plastic toy doll, breast implants, fake tits, "
    "shredded cloth, tattoos, pubic hair"
)
POLL_SECONDS = 1.0
TIMEOUT_SECONDS = 900

SQUARE = (1024, 1024)
PORTRAIT = (896, 1152)
LANDSCAPE = (1152, 896)
WIDE = (1216, 832)

# (style, size, prompt, intended objects, hard objects). Hard objects are easy to miss on purpose.
SCENES = [
    # photograph
    ("photograph", PORTRAIT,
     "Close-up photograph of an elderly fisherman with a weathered face and grey beard, wearing a yellow raincoat, holding a coiled rope in both hands, harbor boats blurred in the background, overcast light, 85mm lens, shallow depth of field.",
     ["old man", "face", "grey beard", "yellow raincoat", "coiled rope", "hands", "boats"], []),
    ("photograph", WIDE,
     "Night street photograph of a group of four friends laughing outside a small bakery, the shop sign above the window reads \"BAKERY\" in glowing neon letters, wet pavement reflecting the lights, a parked red scooter at the curb.",
     ["four people", "bakery storefront", "neon sign reading BAKERY", "wet pavement", "red scooter"], []),
    ("photograph", LANDSCAPE,
     "A golden retriever running along a sandy beach at sunset, splashing through shallow waves, clear orange sky, a lighthouse on a distant rocky point, candid pet photography, fast shutter speed.",
     ["golden retriever", "sandy beach", "waves", "orange sky", "lighthouse", "rocky point"], []),
    ("photograph", SQUARE,
     "Overhead food photograph of a rustic wooden kitchen table with a bowl of ramen, a plate of dumplings, chopsticks, and three lit candles on the table, a small blue teacup in the top corner, warm window light.",
     ["wooden table", "bowl of ramen", "plate of dumplings", "chopsticks", "three lit candles", "blue teacup"],
     ["three lit candles", "blue teacup"]),
    ("photograph", WIDE,
     "Landscape photograph of a pine forest valley at dawn, a calm river winding between the trees, snow-capped mountains behind, no people, soft golden light breaking through the clouds.",
     ["pine trees", "river", "snow-capped mountains", "clouds"], []),
    ("photograph", LANDSCAPE,
     "A brown horse grazing in a green meadow next to a wooden fence, an old red barn and a white farmhouse in the background, blue sky with scattered clouds, summer afternoon, 35mm film look.",
     ["brown horse", "meadow", "wooden fence", "red barn", "white farmhouse", "blue sky", "clouds"], []),
    ("photograph", LANDSCAPE,
     "Interior photograph of a cozy living room, a grey tabby cat sleeping on a green velvet sofa, a bookshelf full of books, a potted fiddle leaf fig, and a small brass telescope standing by the window, afternoon light.",
     ["grey tabby cat", "green sofa", "bookshelf", "books", "potted plant", "brass telescope", "window"],
     ["brass telescope"]),
    ("photograph", WIDE,
     "Urban photograph of a busy city intersection in the rain, yellow taxis and a red double-decker bus waiting at the traffic light, pedestrians with black umbrellas crossing, tall glass office buildings, overcast grey sky.",
     ["yellow taxis", "red double-decker bus", "traffic light", "pedestrians", "black umbrellas", "office buildings", "grey sky"], []),
    ("photograph", SQUARE,
     "Macro photograph of a woman's hands kneading bread dough on a floured marble counter, a wooden rolling pin and a cracked egg beside the dough, soft natural light, no face visible.",
     ["hands", "bread dough", "flour", "marble counter", "rolling pin", "cracked egg"], []),
    # anime illustration
    ("anime illustration", PORTRAIT,
     "Anime illustration of a schoolgirl with short blue hair and a red scarf standing on a train platform at dusk, a commuter train arriving, cherry blossom petals drifting, a station sign reading \"SAKURA\" above her, warm sunset sky.",
     ["girl", "face", "blue hair", "red scarf", "train platform", "train", "cherry blossom petals", "station sign reading SAKURA", "sunset sky"], []),
    ("anime illustration", PORTRAIT,
     "Anime key visual of a young swordsman with silver hair holding a glowing katana inside a ruined stone temple, broken pillars overgrown with vines, dramatic lighting, dynamic pose, detailed cel shading.",
     ["young man", "face", "silver hair", "katana", "stone temple ruins", "broken pillars", "vines"], []),
    ("anime illustration", WIDE,
     "Anime illustration of three friends eating at a small ramen shop counter at night, steam rising from their bowls, paper lanterns hanging overhead, and a small goldfish bowl on the counter, cozy slice of life scene.",
     ["three people", "ramen bowls", "counter", "steam", "paper lanterns", "goldfish bowl"],
     ["goldfish bowl"]),
    ("anime illustration", WIDE,
     "Anime background art of a quiet countryside road running past rice fields, a bus stop with a wooden bench, tall summer clouds, telephone poles, lush green hills, no characters, cinematic lighting.",
     ["road", "rice fields", "bus stop", "wooden bench", "clouds", "telephone poles", "green hills"], []),
    ("anime illustration", SQUARE,
     "Anime illustration of a small red and yellow songbird perched on a branch of a blooming plum tree, snow falling softly, pale grey winter sky, delicate linework, close-up composition.",
     ["songbird", "branch", "plum blossoms", "falling snow", "grey sky"], []),
    ("anime illustration", PORTRAIT,
     "Anime mecha illustration of a giant white robot kneeling in a flooded city street at night, rain pouring, neon signs reflecting in the water, a pilot in a flight suit standing on its open hand.",
     ["giant white robot", "flooded street", "rain", "neon signs", "pilot"], []),
    ("anime illustration", LANDSCAPE,
     "Anime illustration of a witch girl flying on a broom over a seaside town, a black cat riding behind her, a red ribbon in her hair, and a wicker basket of apples hanging from the broom handle, bright blue sky.",
     ["girl", "broom", "seaside town", "sea", "black cat", "red ribbon", "wicker basket of apples", "blue sky"],
     ["wicker basket of apples"]),
    ("anime illustration", WIDE,
     "Anime background of an empty classroom after school, rows of wooden desks, a chalkboard with the word \"EXAM\" written on it, orange sunlight streaming through the windows, no people.",
     ["wooden desks", "chalkboard", "word EXAM", "windows", "sunlight"], []),
    ("anime illustration", SQUARE,
     "Anime illustration of a boy kneeling to hug a fluffy shiba inu dog in a park, both smiling, autumn trees with orange leaves, and a yellow kite tangled in the branches above them.",
     ["boy", "face", "shiba inu dog", "park", "autumn trees", "orange leaves", "yellow kite"],
     ["yellow kite"]),
    # oil painting
    ("oil painting", WIDE,
     "Oil painting of a Dutch harbor at sunset, wooden sailing ships with furled sails, fishermen unloading crates on the dock, golden sky reflected in the calm water, thick impasto brushstrokes, Golden Age style.",
     ["sailing ships", "fishermen", "crates", "dock", "golden sky", "water"], []),
    ("oil painting", PORTRAIT,
     "Classical oil portrait of a noblewoman in a dark green velvet gown, seated with her hands folded in her lap, a pearl necklace, and a small hourglass on the table beside her, dark background, Rembrandt lighting.",
     ["woman", "face", "green velvet gown", "hands", "pearl necklace", "table", "hourglass"],
     ["hourglass"]),
    ("oil painting", LANDSCAPE,
     "Still life oil painting of a copper pot, a loaf of bread, a wedge of cheese, purple grapes and a pheasant on a draped linen tablecloth, dark moody background, Chardin style.",
     ["copper pot", "loaf of bread", "wedge of cheese", "purple grapes", "pheasant", "linen tablecloth"], []),
    ("oil painting", PORTRAIT,
     "Romantic oil painting of an armored knight on a white horse raising a sword, stormy sky with lightning, a castle on a cliff in the distance, dramatic chiaroscuro, sweeping brushwork.",
     ["knight", "armor", "white horse", "sword", "stormy sky", "lightning", "castle", "cliff"], []),
    ("oil painting", WIDE,
     "Impressionist oil painting of a field of sunflowers under a bright summer sky, a row of poplar trees on the horizon, a small stone cottage, visible thick brushstrokes, vibrant yellows and blues.",
     ["sunflowers", "field", "summer sky", "poplar trees", "stone cottage"], []),
    ("oil painting", PORTRAIT,
     "Oil painting of a Parisian cafe terrace at night, people sitting at small round tables, gas lamps glowing, cobblestone street, starry blue night sky, a sign reading \"CAFE\" above the awning.",
     ["people", "round tables", "gas lamps", "cobblestone street", "starry night sky", "sign reading CAFE", "awning"], []),
    ("oil painting", SQUARE,
     "Oil painting of an old wizard reading a large book in a candlelit study, a black cat asleep on stacked books, a stuffed owl on a shelf, and a brass globe on the desk, warm amber tones.",
     ["old wizard", "face", "large book", "candles", "black cat", "stacked books", "stuffed owl", "shelf", "brass globe", "desk"],
     ["stuffed owl", "brass globe"]),
    ("oil painting", LANDSCAPE,
     "Oil painting seascape of towering waves crashing against black rocks, spray and foam, a lone lighthouse beam cutting through grey storm clouds, turbulent brushwork, no people.",
     ["waves", "black rocks", "sea spray", "lighthouse", "storm clouds"], []),
    ("oil painting", LANDSCAPE,
     "Oil painting of ballet dancers rehearsing in a sunlit studio, four girls in white tutus stretching at the barre, a tall mirror on the wall, soft pastel palette, Degas inspired.",
     ["four dancers", "white tutus", "ballet barre", "tall mirror", "studio"], []),
    # 3D render
    ("3D render", SQUARE,
     "3D render of a cute cartoon robot watering a potted cactus on a windowsill with a small watering can, big round glowing eyes, pastel colors, soft global illumination, Pixar style, shallow depth of field.",
     ["robot", "watering can", "potted cactus", "windowsill"], []),
    ("3D render", WIDE,
     "3D render of a vintage red sports car parked on a desert highway at golden hour, chrome details, heat haze, cacti and rocky mesas in the background, clear sky, photorealistic Unreal Engine 5.",
     ["red sports car", "highway", "cacti", "rocky mesas", "clear sky"], []),
    ("3D render", LANDSCAPE,
     "3D render of a minimalist Scandinavian bedroom interior, white bed with grey linen, a wooden nightstand, a monstera plant in the corner, and a yellow alarm clock on the nightstand, soft morning light, architectural visualization.",
     ["bed", "grey linen", "wooden nightstand", "monstera plant", "yellow alarm clock"],
     ["yellow alarm clock"]),
    ("3D render", PORTRAIT,
     "3D render of a stylized fantasy warrior woman holding a large broadsword in both hands, braided red hair, leather armor, standing on a mossy cliff edge, game character concept art, Blender Cycles.",
     ["woman", "face", "broadsword", "hands", "braided red hair", "leather armor", "mossy cliff"], []),
    ("3D render", SQUARE,
     "3D render of a stack of fluffy pancakes with melting butter and maple syrup dripping, fresh blueberries and strawberries on top, on a white plate, studio lighting, hyperrealistic food visualization.",
     ["pancakes", "butter", "maple syrup", "blueberries", "strawberries", "white plate"], []),
    ("3D render", PORTRAIT,
     "3D render of a futuristic cyberpunk city at night, flying cars between skyscrapers, holographic billboards, a giant neon sign reading \"NOODLES\", rain-slicked streets, cinematic lighting.",
     ["flying cars", "skyscrapers", "holographic billboards", "neon sign reading NOODLES", "wet streets"], []),
    ("3D render", LANDSCAPE,
     "3D render of a family of three penguins sliding across an ice floe in the Antarctic sea, snowy peaks behind them, bright blue sky, stylized animation film look, soft lighting.",
     ["three penguins", "ice floe", "sea", "snowy peaks", "blue sky"], []),
    ("3D render", WIDE,
     "3D render of tiny clay-style villagers building a snowman in a village square, cottages with snowy roofs, and a green bicycle leaning against a lamppost, soft toy-like lighting, tilt-shift look.",
     ["villagers", "snowman", "village square", "cottages", "snowy roofs", "green bicycle", "lamppost"],
     ["green bicycle"]),
    ("3D render", PORTRAIT,
     "3D render of a glass perfume bottle on a black marble pedestal, surrounded by floating rose petals, dramatic rim lighting, dark studio background, luxury product shot.",
     ["glass perfume bottle", "black marble pedestal", "rose petals"], []),
    # watercolor
    ("watercolor", PORTRAIT,
     "Watercolor painting of a Venetian canal with gondolas, pastel colored buildings with laundry lines, a gondolier in a striped shirt, soft reflections in the water, loose wet-on-wet washes.",
     ["canal", "gondolas", "pastel buildings", "laundry lines", "gondolier", "water"], []),
    ("watercolor", SQUARE,
     "Watercolor illustration of a blue kingfisher diving into a pond to catch a fish, lily pads and reeds, splashes of water, white paper background showing through, botanical art style.",
     ["kingfisher", "fish", "pond", "lily pads", "reeds", "water splashes"], []),
    ("watercolor", PORTRAIT,
     "Watercolor painting of a little girl in a yellow raincoat jumping in puddles on a village street, rainy grey sky, and a red umbrella lying open on the pavement behind her, soft bleeding colors.",
     ["girl", "yellow raincoat", "puddles", "village street", "grey sky", "red umbrella"],
     ["red umbrella"]),
    ("watercolor", LANDSCAPE,
     "Watercolor recipe illustration of a fresh summer salad bowl with tomatoes, cucumber, black olives and feta cheese, a bottle of olive oil beside it, loose ink outlines, light cheerful palette.",
     ["salad bowl", "tomatoes", "cucumber", "black olives", "feta cheese", "olive oil bottle"], []),
    ("watercolor", PORTRAIT,
     "Watercolor landscape of autumn birch trees along a forest path, golden leaves scattered on the ground, a small wooden footbridge over a stream, soft morning light, no people.",
     ["birch trees", "forest path", "golden leaves", "wooden footbridge", "stream"], []),
    ("watercolor", LANDSCAPE,
     "Watercolor portrait of a sleepy dachshund curled up in a knitted basket by a fireplace, cozy interior, warm glow from the fire, a pair of slippers on the rug.",
     ["dachshund", "knitted basket", "fireplace", "fire", "slippers", "rug"], []),
    ("watercolor", SQUARE,
     "Watercolor sketch of a flower shop storefront with buckets of tulips and roses outside, the sign above the door reads \"FLORIST\", a bicycle with a basket parked out front, and a birdcage hanging in the window.",
     ["flower shop", "tulips", "roses", "buckets", "sign reading FLORIST", "bicycle", "birdcage"],
     ["birdcage"]),
    ("watercolor", LANDSCAPE,
     "Watercolor painting of a family of four sitting around a campfire at night, roasting marshmallows on sticks, a tent behind them, tall pine trees, and a starry sky with a crescent moon.",
     ["four people", "campfire", "marshmallows", "tent", "pine trees", "starry sky", "crescent moon"], []),
    ("watercolor", WIDE,
     "Watercolor painting of wild horses galloping across an open plain, dust rising under their hooves, distant blue mountains, loose expressive brushwork, muted earth tones.",
     ["horses", "plain", "dust", "blue mountains"], []),
    # pencil sketch
    ("pencil sketch", PORTRAIT,
     "Pencil sketch portrait of an old man with deep wrinkles and a flat cap, smoking a pipe, detailed cross-hatching, graphite on textured paper, realistic shading.",
     ["old man", "face", "wrinkles", "flat cap", "pipe"], []),
    ("pencil sketch", SQUARE,
     "Detailed pencil drawing study of two hands, one holding a pen and the other resting on an open notebook, graphite shading, anatomical sketchbook page.",
     ["two hands", "pen", "open notebook"], []),
    ("pencil sketch", PORTRAIT,
     "Pencil sketch of a gothic cathedral facade with a rose window and tall spires, pigeons on the front steps, a few pedestrians walking past, architectural drawing, fine linework.",
     ["gothic cathedral", "rose window", "spires", "pigeons", "steps", "pedestrians"], []),
    ("pencil sketch", LANDSCAPE,
     "Pencil sketch of a cat sitting on a cluttered writer's desk beside a typewriter, a stack of letters, and a pair of round spectacles resting on the typewriter keys, soft graphite shading.",
     ["cat", "desk", "typewriter", "stack of letters", "round spectacles"],
     ["round spectacles"]),
    ("pencil sketch", WIDE,
     "Pencil sketch of a steam locomotive crossing a stone viaduct over a river valley, billowing steam, detailed mechanical linework, trees on the hillsides.",
     ["steam locomotive", "stone viaduct", "river", "steam", "trees"], []),
    ("pencil sketch", PORTRAIT,
     "Graphite pencil drawing of an owl perched on a fence post at night, a full moon behind it, and a small mouse hiding in the grass below, detailed feather texture.",
     ["owl", "fence post", "full moon", "small mouse", "grass"],
     ["small mouse"]),
    ("pencil sketch", WIDE,
     "Pencil sketch of a crowded outdoor market, vendors selling fruit from wooden carts, shoppers carrying baskets, striped awnings, a small dog weaving between legs, loose reportage drawing.",
     ["market", "vendors", "fruit", "wooden carts", "shoppers", "baskets", "awnings", "dog"], []),
    ("pencil sketch", LANDSCAPE,
     "Pencil concept sketch of an ornate fantasy sword with a dragon-shaped hilt, lying on a stone altar, handwritten annotations in the margins, graphite with white highlights.",
     ["sword", "dragon-shaped hilt", "stone altar", "annotations"], []),
    ("pencil sketch", WIDE,
     "Pencil landscape sketch of a mountain lake with a small rowboat tied to a wooden jetty, pine trees on the shore, reflections on the water, soft cloudy sky.",
     ["mountain lake", "rowboat", "wooden jetty", "pine trees", "water", "cloudy sky"], []),
    # pixel art
    ("pixel art", SQUARE,
     "Pixel art of a knight with a sword and shield standing before a dungeon gate, torches burning on the stone walls, 16-bit retro game style, limited color palette.",
     ["knight", "sword", "shield", "dungeon gate", "torches", "stone walls"], []),
    ("pixel art", PORTRAIT,
     "Pixel art of a rainy city street at night, a ramen stand with a glowing sign reading \"OPEN\", a person holding an umbrella, neon reflections on the wet ground, 32-bit style.",
     ["city street", "ramen stand", "sign reading OPEN", "person", "umbrella", "neon reflections"], []),
    ("pixel art", LANDSCAPE,
     "Pixel art farm scene, a red barn, a farmer watering crops, chickens in a pen, a brown horse by the fence, and a blue wheelbarrow near the gate, sunny day, cozy farming game style.",
     ["red barn", "farmer", "crops", "chickens", "brown horse", "fence", "blue wheelbarrow", "gate"],
     ["blue wheelbarrow"]),
    ("pixel art", WIDE,
     "Pixel art landscape of a waterfall pouring into a jungle lagoon, palm trees, bright blue sky with fluffy clouds, parallax background layers, vibrant 16-bit palette, no characters.",
     ["waterfall", "lagoon", "palm trees", "blue sky", "clouds"], []),
    ("pixel art", WIDE,
     "Pixel art of a spaceship flying through an asteroid field, a large ringed planet in the background, stars, laser blasts, retro arcade shooter style.",
     ["spaceship", "asteroids", "ringed planet", "stars", "laser blasts"], []),
    ("pixel art", SQUARE,
     "Pixel art interior of a cozy bakery, shelves of bread loaves and croissants, a baker behind the counter, a layered cake on display, warm lighting, isometric view.",
     ["shelves", "bread loaves", "croissants", "baker", "counter", "cake"], []),
    ("pixel art", LANDSCAPE,
     "Pixel art of a cat sleeping on a windowsill in a bedroom at night, the moon visible through the window, a lava lamp on the desk, and two potted succulents on the sill, lofi aesthetic.",
     ["cat", "windowsill", "window", "moon", "lava lamp", "desk", "two potted succulents"],
     ["lava lamp", "two potted succulents"]),
    ("pixel art", SQUARE,
     "Pixel art character portrait of a smiling elf girl with green hair, pointed ears and a flower crown, RPG dialogue portrait, detailed sprite shading.",
     ["elf girl", "face", "green hair", "pointed ears", "flower crown"], []),
    ("pixel art", LANDSCAPE,
     "Pixel art of a party of four adventurers around a tavern table, mugs of ale, a roast chicken, a bard playing a lute, candle light, RPG town style.",
     ["four adventurers", "tavern table", "mugs", "roast chicken", "bard", "lute", "candles"], []),
    # comic book
    ("comic book", PORTRAIT,
     "Comic book panel of a superhero in a blue and silver suit landing on a rooftop at night, cape flowing, city skyline behind, bold ink lines, halftone shading, dynamic angle.",
     ["superhero", "face", "cape", "rooftop", "city skyline", "night sky"], []),
    ("comic book", PORTRAIT,
     "Comic book cover illustration of a detective in a trench coat and fedora standing in a dark alley, a speech bubble saying \"WHO DID IT?\", falling rain, noir ink style.",
     ["detective", "trench coat", "fedora", "alley", "speech bubble reading WHO DID IT?", "rain"], []),
    ("comic book", WIDE,
     "Comic book panel of a car chase through a city street, a yellow taxi swerving past a fire hydrant, and a hot dog cart knocked over with sausages flying, sound effect lettering \"SCREECH\", bold colors.",
     ["yellow taxi", "city street", "fire hydrant", "hot dog cart", "sausages", "lettering SCREECH"],
     ["fire hydrant", "hot dog cart"]),
    ("comic book", LANDSCAPE,
     "Comic strip illustration of a scruffy dog and a grumpy cat sharing a sofa, a bowl of popcorn between them, a TV glowing in front, simple bold outlines, flat colors, newspaper comic look.",
     ["dog", "cat", "sofa", "bowl of popcorn", "TV"], []),
    ("comic book", PORTRAIT,
     "Comic book splash page of a samurai and a ninja clashing swords on a bamboo forest path, motion lines, sparks, dramatic ink shading, manga-influenced American comic style.",
     ["samurai", "ninja", "swords", "bamboo forest", "path", "sparks"], []),
    ("comic book", SQUARE,
     "Comic book illustration of a mad scientist in a lab coat in his laboratory, bubbling green flasks, a chalkboard of equations, and a rubber duck sitting on top of the control panel, ink and flat colors.",
     ["scientist", "lab coat", "green flasks", "chalkboard", "rubber duck", "control panel"],
     ["rubber duck"]),
    ("comic book", WIDE,
     "Comic book landscape of a desert canyon at noon, red rock formations, a winding dirt road, a vulture circling in the bright sky, bold outlines and Ben-Day dots, no characters.",
     ["desert canyon", "red rock formations", "dirt road", "vulture", "sky"], []),
    ("comic book", LANDSCAPE,
     "Comic book panel of a family of five at a dinner table celebrating a birthday, a cake with lit candles, balloons tied to the chairs, everyone cheering, bright pop art colors.",
     ["five people", "dinner table", "birthday cake", "lit candles", "balloons", "chairs"], []),
    ("comic book", WIDE,
     "Comic book panel of pirates on a wooden ship fighting a giant octopus in a stormy sea, one pirate swinging a cutlass, and a parrot wearing a tiny hat on the captain's shoulder, bold inks, dramatic lightning.",
     ["pirates", "wooden ship", "giant octopus", "stormy sea", "cutlass", "parrot", "tiny hat", "lightning"],
     ["parrot", "tiny hat"]),
]


def build_graph(prompt: str, width: int, height: int, seed: int, steps: int, prefix: str) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": NEGATIVE}},
        "guider": {"class_type": "CFGGuider",
                   "inputs": {"model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": CFG}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": ["unet", 0], "scheduler": SCHEDULER, "steps": steps, "denoise": 1.0}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                              "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


def http_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def run_prompt(graph: dict) -> tuple[Path, float]:
    """Queue one graph, wait for it, return the saved PNG and its execution seconds."""
    prompt_id = http_json("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})["prompt_id"]
    started = time.monotonic()
    entry = None
    while time.monotonic() - started < TIMEOUT_SECONDS:
        entry = http_json(f"/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            break
        time.sleep(POLL_SECONDS)
    wall_seconds = time.monotonic() - started
    if not entry or entry["status"].get("status_str") != "success":
        raise RuntimeError(f"prompt {prompt_id} failed or timed out: {json.dumps(entry)[:2000] if entry else 'no history'}")

    # The server's own start and success timestamps exclude queue and poll latency.
    stamps = {name: body.get("timestamp") for name, body in entry["status"].get("messages", [])}
    seconds = wall_seconds
    if stamps.get("execution_start") and stamps.get("execution_success"):
        seconds = (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
    image = entry["outputs"]["save"]["images"][0]
    return OUTPUT_DIR / image["subfolder"] / image["filename"], seconds


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def write_manifest(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda record: record["id"])
    MANIFEST.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_scenes() -> None:
    for index, (style, _, prompt, intended, hard) in enumerate(SCENES, start=1):
        words = len(prompt.split())
        if not 20 <= words <= 60:
            raise ValueError(f"g{index:02d} ({style}) has {words} words, outside 20-60")
        missing = [item for item in hard if item not in intended]
        if missing:
            raise ValueError(f"g{index:02d} hard objects {missing} are not in intended")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--only", default="", help="comma separated ids, e.g. g01,g02")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new renders")
    args = parser.parse_args()

    validate_scenes()
    only_ids = {item.strip() for item in args.only.split(",") if item.strip()}
    records = {record["id"]: record for record in load_manifest()}
    rendered_count = 0
    failures = []

    for index, (style, (width, height), prompt, intended, hard) in enumerate(SCENES, start=1):
        image_id = f"g{index:02d}"
        if only_ids and image_id not in only_ids:
            continue
        if image_id in records and (RENDERS_DIR / records[image_id]["file"]).exists() and not only_ids:
            continue
        if args.limit and rendered_count >= args.limit:
            break
        seed = SEED_BASE + index
        graph = build_graph(prompt, width, height, seed, args.steps, f"LT-testset/{image_id}")
        try:
            source, seconds = run_prompt(graph)
        except Exception as error:  # one bad render must not stop the set
            failures.append(f"{image_id}: {error}")
            print(f"{image_id} FAILED {error}", flush=True)
            continue
        copy_path = RENDERS_DIR / f"{image_id}.png"
        shutil.copy2(source, copy_path)
        records[image_id] = {
            "id": image_id, "file": copy_path.name, "prompt": prompt, "style": style,
            "intended": intended, "hard": hard, "width": width, "height": height, "seed": seed,
            "steps": args.steps, "seconds": round(seconds, 2),
        }
        write_manifest(list(records.values()))
        rendered_count += 1
        print(f"{image_id} {style} {width}x{height} {seconds:.1f}s", flush=True)

    print(f"rendered {rendered_count}, failed {len(failures)}, manifest has {len(records)}")
    for failure in failures:
        print("  " + failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
