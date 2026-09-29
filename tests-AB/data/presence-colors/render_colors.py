"""Render the presence colors set: Krea 2 renders that ask 12 nouns which also name a color.

A tag such as "orange" names the fruit to a tagger, but a wording without an article can read it as the color. Object
scenes hold the thing in neutral colors, color scenes hold only the color ("an orange coat"), and mixed scenes hold
one word's thing beside another word's color. Every render asks all 12 words, and judges label each by its thing
reading (../presence-forms/JUDGE.md). Rendering reuses ../presence-forms/render_forms.py.

    python render_colors.py [--only c01,c02] [--limit N]
    python ../presence-forms/labels.py split --dir .
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED_BASE = 4000
WORDS = ("orange", "lime", "lemon", "peach", "plum", "cherry", "olive", "rose", "lavender", "gold", "chocolate", "mint")

_spec = importlib.util.spec_from_file_location("render_forms", HERE.parent / "presence-forms" / "render_forms.py")
render_forms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_forms)
render = render_forms.render

# (style, size, prompt, words whose thing the prompt names)
COLOR_SCENES = [
    ("photograph", render.LANDSCAPE,
     "Still life photograph of three oranges and two limes on a plain grey stone table, soft window light, neutral "
     "grey background, sharp focus, realistic detail.", ("orange", "lime")),
    ("photograph", render.LANDSCAPE,
     "Photograph of a whole lemon and a ripe peach side by side on a white marble counter, soft daylight, minimal "
     "white background, sharp focus, realistic detail.", ("lemon", "peach")),
    ("photograph", render.SQUARE,
     "Photograph of a small white bowl of dark plums and red cherries on a grey linen cloth, overcast window light, "
     "neutral background, sharp focus, realistic detail.", ("plum", "cherry")),
    ("photograph", render.SQUARE,
     "Photograph of a glass jar of green olives beside a single red rose lying on a grey wooden table, muted light, "
     "neutral background, sharp focus.", ("olive", "rose")),
    ("photograph", render.LANDSCAPE,
     "Photograph of a bundle of dried lavender tied with string next to a shiny gold bar on a black slate surface, "
     "studio lighting, dark background, sharp focus.", ("lavender", "gold")),
    ("photograph", render.SQUARE,
     "Photograph of a broken bar of dark chocolate and a sprig of fresh mint leaves on a white ceramic plate, grey "
     "background, soft light, sharp focus.", ("chocolate", "mint")),
    ("photograph", render.PORTRAIT,
     "Fashion photograph of a woman in a bright orange wool coat walking past a lime green painted door on a grey "
     "city street, overcast light, no fruit.", ()),
    ("photograph", render.LANDSCAPE,
     "Interior photograph of a clean empty kitchen with lemon yellow cabinets and peach colored walls, a white sink, "
     "no food anywhere, soft daylight.", ()),
    ("photograph", render.WIDE,
     "Street photograph of a vintage car painted plum purple parked beside a cherry red scooter on an empty cobbled "
     "street, afternoon light, no people.", ()),
    ("photograph", render.PORTRAIT,
     "Portrait photograph of a man in an olive green jacket holding a rose pink umbrella in light rain, blurred city "
     "background, soft light, no flowers.", ()),
    ("photograph", render.LANDSCAPE,
     "Interior photograph of a bedroom with lavender painted walls, a gold colored picture frame and a white bed "
     "with neat sheets, soft morning light.", ()),
    ("photograph", render.LANDSCAPE,
     "Interior photograph of a living room with a chocolate brown leather sofa and mint green cushions against plain "
     "white walls, soft daylight, no plants.", ()),
    ("photograph", render.PORTRAIT,
     "Photograph of a woman in an orange knitted sweater peeling a lemon at a white kitchen table, soft window light, "
     "shallow depth of field.", ("lemon",)),
    ("photograph", render.LANDSCAPE,
     "Photograph of a lime green bicycle leaning against a white wall beside a wooden crate full of ripe peaches, "
     "morning light, sharp focus.", ("peach",)),
    ("anime illustration", render.PORTRAIT,
     "Anime illustration of a girl in a plum colored dress holding a small wicker basket of red cherries in a sunny "
     "garden, soft linework, pastel background.", ("cherry",)),
    ("photograph", render.LANDSCAPE,
     "Photograph of an olive green tent at a forest campsite with a small pot of melted chocolate warming over a "
     "campfire, evening light.", ("chocolate",)),
    ("photograph", render.WIDE,
     "Photograph of a lavender colored vintage car parked on a quiet street with a bouquet of red roses lying on its "
     "hood, soft afternoon light.", ("rose",)),
    ("photograph", render.WIDE,
     "Photograph of a mint green shop front with a white bowl of oranges in the window, a black bicycle parked "
     "outside, overcast light, sharp focus.", ("orange",)),
]


def scenes() -> list:
    """COLOR_SCENES in render_forms' scene shape: the named words as intended, every other word as a distractor."""
    return [(style, size, prompt, dict.fromkeys(named, "color-word"),
             {word: "color-word" for word in WORDS if word not in named})
            for style, size, prompt, named in COLOR_SCENES]


if __name__ == "__main__":
    sys.exit(render_forms.render_set(scenes(), HERE, "c", SEED_BASE))
