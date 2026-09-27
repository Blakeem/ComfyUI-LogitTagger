"""List inputs turned into items: one image per item, texts paired to the items, one widget value.

A node with INPUT_IS_LIST = True receives every input as a list, widgets included.
Module scope is stdlib only.
"""

from __future__ import annotations

from typing import Any

from . import engine


def split_images(images: list[Any], node: str) -> list[Any]:
    """Every image of every IMAGE batch in images, in order, each as its own [1, H, W, C] tensor."""
    items: list[Any] = []

    for batch in images:
        engine.check_images(batch, node)
        items.extend(batch[index:index + 1] for index in range(batch.shape[0]))
    return items


def pair_texts(texts: list[str] | None, count: int, node: str, input_name: str) -> list[str]:
    """One text per image: none gives empty texts, one text repeats, count texts pair one to one."""
    values = list(texts or [])

    if not values:
        return [""] * count
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return values
    raise ValueError(
        f"{node}: the {input_name} input has {len(values)} texts for {count} images. Connect one text "
        f"to use for every image, or {count} texts to pair one to one with the images."
    )


def single_value(values: list[Any], node: str, input_name: str) -> Any:
    """The one value of a widget that INPUT_IS_LIST delivers as a list."""
    distinct: list[str] = []

    if not values:
        raise ValueError(f"{node}: the {input_name} input received no value. Set the {input_name} widget.")
    if any(value != values[0] for value in values[1:]):
        distinct = list(dict.fromkeys(repr(value) for value in values))
        raise ValueError(
            f"{node}: the {input_name} input received {len(distinct)} different values "
            f"({', '.join(distinct)}). It takes one value for the whole run, so connect a single value."
        )
    return values[0]
