"""Check the prompt stage's language gate and translation over a real Qwen3-VL-4B CLIP, with no server.

Each prompt prints the gate's p(English), the text the listing reads and the things prompt_things keeps. The English
set holds style-only prompts, since those list no things and so look foreign to word grounding alone.
The output sets LANGUAGE_QUESTION, ENGLISH_THRESHOLD and TRANSLATE_INSTRUCTION in logit_tagger/tagging.py.
Usage: python tests-AB/ab_prompt_language.py [--clip NAME]
"""

import argparse
import time

import ab_env
from ab_tagger import DEFAULT_CLIP

FOREIGN = (
    "Eine Katze liegt auf einer Computertastatur.",
    "Ein alter Mann mit rotem Hut sitzt auf einer Holzbank im Park, neben ihm ein brauner Hund und eine Laterne.",
    "Une femme en robe bleue tient un parapluie rouge sous la pluie, devant un café avec des chaises en fer.",
    "Un gato naranja duerme sobre una manta de lana junto a una ventana con flores.",
    "赤い着物を着た女性が桜の木の下で傘を持って立っている",
    "一只白色的狗在海滩上奔跑，背景是日落和棕榈树",  # noqa: RUF001
    "Uma menina com vestido amarelo brinca com um cachorro no jardim.",
    "Кот сидит на подоконнике рядом с цветком.",  # noqa: RUF001
    "Een man op een fiets rijdt langs een gracht met bootjes.",
    "Una ragazza con un cappello di paglia mangia un gelato in piazza.",
    "filmisch, stimmungsvoll, 35mm Film, goldene Stunde",
    "chef-d'oeuvre, meilleure qualité, ultra détaillé, 8k",
    "cinematic photo, eine Katze auf einem Sofa, soft light",
)
ENGLISH = (
    "cinematic, moody, 35mm film, golden hour",
    "masterpiece, best quality, ultra detailed, 8k, sharp focus",
    "1girl, solo, long hair, smile, looking at viewer",
    "A cat lying on a computer keyboard.",
    "A photo of two cats and a keybaord.",
    "dreamy pastel aesthetic, soft bokeh, film grain, vaporwave vibes",
    "epic fantasy concept art by greg rutkowski, trending on artstation",
    "portrait, dramatic lighting, rembrandt style",
    "an abstract composition of pure color and motion",
    "cozy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--clip", default=DEFAULT_CLIP, help="text encoder file name under models/clip")
    return parser.parse_args()


def main() -> None:
    ab_env.bootstrap()
    from logit_tagger import engine, tagging

    args = parse_args()
    clip = ab_env.load_clip(args.clip)
    classifier = engine.build_classifier(clip, tagging.PROMPT_CHECK_NAME, needs_images=False)
    gate: dict[str, list[float]] = {"foreign": [], "english": []}
    p_english = 0.0
    text = ""
    things: tuple[str, ...] = ()
    started = 0.0

    # The first generate pays the model load and graph capture, so it runs once untimed.
    tagging.prompt_things(clip, classifier, ENGLISH[3], {})
    ab_env.end_node_execution()

    for group, prompts in (("foreign", FOREIGN), ("english", ENGLISH)):
        for prompt in prompts:
            started = time.perf_counter()
            p_english = engine.nouls(classifier, [tagging.LANGUAGE_QUESTION], state=prompt)[0]
            text = tagging.english_prompt(clip, classifier, prompt)
            things = tagging.prompt_things(clip, classifier, prompt, {})
            ab_env.end_node_execution()
            gate[group].append(p_english)
            print(f"{group:7} p(English) {p_english:.3f} {(time.perf_counter() - started):5.2f} s  {prompt[:60]!r}\n"
                  f"        read as {text[:100]!r}\n        things {things}", flush=True)

    print(f"foreign p(English) {min(gate['foreign']):.3f} to {max(gate['foreign']):.3f}, "
          f"english {min(gate['english']):.3f} to {max(gate['english']):.3f}, threshold {tagging.ENGLISH_THRESHOLD}")


if __name__ == "__main__":
    main()
