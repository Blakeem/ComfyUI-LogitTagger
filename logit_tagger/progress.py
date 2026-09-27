"""One progress bar and one status line for a node run over a list of items.

Core's clip.generate builds its own comfy.utils.ProgressBar per call (comfy/text_encoders/llama.py:1154),
and each new bar resets the node's display. Inside `with run:` the toolkit's routed_progress_bars folds
those bars into the run's one bar. Module scope is stdlib only, so comfy, the server and the library are
imported inside the functions that need them.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Any

from . import engine

# The run's bar carries integers, so this is its resolution per item.
ITEM_UNITS = 100
# Core sizes its token bar to the token budget and a greedy reply stops early, as ContextAnchoredTileRefine measured.
GENERATE_FILL_RATIO = 0.65
# The generate's share of one item. The next item's start, or a clean exit, fills the rest.
GENERATE_SHARE = 0.8


class RunProgress:
    """The run's one ProgressBar over total_items items, and the status line under it.

    The emitted value never decreases and the emitted total never drops below it.
    """

    def __init__(self, total_items: int, unique_id: str | None, noun: str, *, node: str) -> None:
        # The Tagger and Prompt Check nodes enter the run before build_classifier checks the library.
        try:
            import logit_classifier.toolkit.comfyui as toolkit
        except ImportError as error:
            raise engine.library_error(node) from error
        import comfy.utils

        self._toolkit = toolkit
        self._routing = ExitStack()
        self._total_items = int(total_items)
        self._unique_id = unique_id
        self._noun = noun
        self._index: int | None = None
        self._value = 0
        self._total = self._total_items * ITEM_UNITS
        self._status_text: str | None = None
        self._started = time.perf_counter()
        # Built before __enter__ routes comfy.utils.ProgressBar, so the run's own bar is core's class.
        self._bar = comfy.utils.ProgressBar(self._total)
        self._emit(0)

    def item(self, index: int, stage: str) -> None:
        """Make item index current and show stage for it, after core's interrupt check."""
        import comfy.model_management

        text = ""

        comfy.model_management.throw_exception_if_processing_interrupted()
        self._index = int(index)
        text = stage if self._total_items == 1 else f"{self._noun} {self._index + 1}/{self._total_items}: {stage}"
        self._emit(self._index * ITEM_UNITS)
        self._send_status(text)

    def route(self, value: float, total: float) -> None:
        """Map a routed bar's own progress into the current item's generate share."""
        fill = 0.0

        # Before the first item there is no share to fill.
        if self._index is None:
            return
        fill = self._toolkit.token_fill(value, total, GENERATE_FILL_RATIO)
        self._emit((self._index + GENERATE_SHARE * fill) * ITEM_UNITS)

    def __enter__(self) -> RunProgress:
        self._routing.enter_context(
            self._toolkit.routed_progress_bars(lambda value, total, _preview: self.route(value, total)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        import comfy.model_management

        seconds = time.perf_counter() - self._started
        noun = self._noun if self._total_items == 1 else f"{self._noun}s"
        text = ""

        self._routing.close()
        text = self._toolkit.run_outcome(exc_type, f"{self._total_items} {noun} in {seconds:.1f} s")
        if exc_type is None:
            # Core's bar hook checks for Cancel on the final update, so the fill can raise the run's interrupt.
            try:
                self._emit(self._total_items * ITEM_UNITS)
            except comfy.model_management.InterruptProcessingException:
                self._send_status("cancelled")
                raise
        self._send_status(text)
        return False

    def _emit(self, units: float) -> None:
        self._value = max(self._value, round(units))
        self._total = max(self._total, self._value)
        self._bar.update_absolute(self._value, self._total)

    def _send_status(self, text: str) -> None:
        if text == self._status_text:
            return
        self._status_text = text
        self._toolkit.send_status(self._unique_id, text)
