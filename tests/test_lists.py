"""lists.py: the rules that turn INPUT_IS_LIST inputs into one item per image.

Plain tensors stand in for ComfyUI IMAGE batches, so no comfy install and no GPU are needed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from logit_tagger import lists

PACK_ROOT = Path(__file__).parent.parent.resolve()
NODE = "Some Node"


def a_batch(batch=1, height=8, width=8, fill=0.0):
    torch = pytest.importorskip("torch")
    return torch.full((batch, height, width, 3), fill)


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_no_torch_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.lists\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- split_images -----------------------------------------------------------------------


def test_split_images_gives_every_image_of_every_batch_in_order_as_its_own_batch():
    first = a_batch(batch=2, height=8, width=8, fill=1.0)
    second = a_batch(batch=1, height=16, width=12, fill=2.0)

    images = lists.split_images([first, second], NODE)

    assert [tuple(image.shape) for image in images] == [(1, 8, 8, 3), (1, 8, 8, 3), (1, 16, 12, 3)]
    assert images[0].equal(first[0:1])
    assert images[1].equal(first[1:2])
    assert images[2].equal(second)


def test_split_images_of_an_empty_list_is_empty():
    assert lists.split_images([], NODE) == []


def test_split_images_rejects_a_batch_that_is_not_an_image_naming_the_node():
    with pytest.raises(ValueError, match=NODE):
        lists.split_images([a_batch(), a_batch()[0]], NODE)


# --- pair_texts -------------------------------------------------------------------------


@pytest.mark.parametrize("texts", [None, []])
def test_pair_texts_without_texts_gives_one_empty_text_per_item(texts):
    assert lists.pair_texts(texts, 3, NODE, "prompt") == ["", "", ""]


def test_pair_texts_repeats_one_text_for_every_item():
    assert lists.pair_texts(["a cat"], 3, NODE, "prompt") == ["a cat", "a cat", "a cat"]


def test_pair_texts_pairs_count_texts_one_to_one():
    assert lists.pair_texts(["a cat", "a dog", "a fox"], 3, NODE, "prompt") == ["a cat", "a dog", "a fox"]


def test_pair_texts_rejects_any_other_length_naming_the_node_the_input_both_lengths_and_the_rule():
    with pytest.raises(ValueError) as error:
        lists.pair_texts(["a cat", "a dog"], 3, NODE, "prompt")

    message = str(error.value)
    assert NODE in message
    assert "prompt" in message
    assert "2 texts" in message
    assert "3 items" in message
    assert "one text to use for every item" in message


# --- single_value -----------------------------------------------------------------------


def test_single_value_returns_the_value_of_a_one_entry_list():
    assert lists.single_value([0.5], NODE, "threshold") == 0.5


def test_single_value_returns_the_value_when_every_entry_is_equal():
    assert lists.single_value(["fast", "fast", "fast"], NODE, "mode") == "fast"


def test_single_value_rejects_entries_that_differ_naming_the_node_and_the_input():
    with pytest.raises(ValueError) as error:
        lists.single_value([0.5, 0.7], NODE, "threshold")

    message = str(error.value)
    assert NODE in message
    assert "threshold" in message
    assert "0.5" in message
    assert "0.7" in message


def test_single_value_rejects_an_empty_list_naming_the_node_and_the_input():
    with pytest.raises(ValueError) as error:
        lists.single_value([], NODE, "threshold")

    assert NODE in str(error.value)
    assert "threshold" in str(error.value)
