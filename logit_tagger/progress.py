"""One progress bar and one status line for a node run over a list of items.

Core's clip.generate builds its own comfy.utils.ProgressBar per call (comfy/text_encoders/llama.py:1154),
and each new bar resets the node's display. Inside `with run:` those bars route into the run's one bar.
Module scope is stdlib only, so comfy and the server are imported inside the functions that need them.
"""

from __future__ import annotations

import time
from typing import Any

# The run's bar carries integers, so this is its resolution per item.
ITEM_UNITS = 100
# Core sizes its token bar to the token budget and a greedy reply stops early, as ContextAnchoredTileRefine measured.
GENERATE_FILL_RATIO = 0.65
# The generate's share of one item. The next item's start, or a clean exit, fills the rest.
GENERATE_SHARE = 0.8


def send_status(unique_id: str | None, text: str) -> None:
    """Show text under the node's progress bar, or do nothing without a server, an instance or a node id."""
    instance: Any = None

    if unique_id is None:
        return
    try:
        from server import PromptServer
    except ImportError:
        return
    # Core sets PromptServer.instance only when it constructs a server.
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return
    try:
        instance.send_progress_text(text, unique_id)
    except Exception:
        # The status line is decoration, so a failed send must never fail the run.
        return


def generate_fill(value: float, total: float) -> float:
    """A token bar's progress as a fraction of the generate share, held at 1 once it reaches the fill ratio."""
    if total <= 0:
        return 1.0
    return min(1.0, max(0.0, value / (GENERATE_FILL_RATIO * total)))


class RunProgress:
    """The run's one ProgressBar over total_items items, and the status line under it.

    The emitted value never decreases and the emitted total never drops below it.
    """

    def __init__(self, total_items: int, unique_id: str | None, noun: str) -> None:
        import comfy.utils

        # Captured before __enter__ patches the attribute, so the run's own bar is core's class.
        self._bar_class = comfy.utils.ProgressBar
        self._total_items = int(total_items)
        self._unique_id = unique_id
        self._noun = noun
        self._index: int | None = None
        self._value = 0
        self._total = self._total_items * ITEM_UNITS
        self._status_text: str | None = None
        self._started = time.perf_counter()
        self._bar = self._bar_class(self._total)
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
        units = 0.0

        # Before the first item there is no share to fill.
        if self._index is None:
            return
        units = (self._index + GENERATE_SHARE * generate_fill(value, total)) * ITEM_UNITS
        self._emit(units)

    def __enter__(self) -> RunProgress:
        import comfy.utils

        comfy.utils.ProgressBar = _routed_bar_class(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        import comfy.model_management
        import comfy.utils

        seconds = time.perf_counter() - self._started
        noun = self._noun if self._total_items == 1 else f"{self._noun}s"
        text = f"{self._total_items} {noun} in {seconds:.1f} s"

        comfy.utils.ProgressBar = self._bar_class
        if exc_type is not None and issubclass(exc_type, comfy.model_management.InterruptProcessingException):
            text = "cancelled"
        elif exc_type is not None:
            text = "failed"
        else:
            # Core's bar hook checks for Cancel on the final update, so the fill can raise the run's interrupt.
            try:
                self._emit(self._total_items * ITEM_UNITS)
            except comfy.model_management.InterruptProcessingException:
                self._send_status("cancelled")
                raise
        # Frontend 1.53.6 keeps the last progress text and ignores an empty one, so the run ends on a line that
        # says how it ended.
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
        send_status(self._unique_id, text)


def _routed_bar_class(run: RunProgress) -> type:
    """comfy.utils.ProgressBar's construction and update surface, routed into run."""
    import comfy.model_management

    # A class per run, so nothing outlives the patch window. Core passes node_id and preview in a few places.
    class RoutedProgressBar:
        def __init__(self, total: float, node_id: Any = None) -> None:
            self.total = total
            self.current = 0
            self.node_id = node_id

        def update_absolute(self, value: float, total: float | None = None, preview: Any = None) -> None:
            # Core's bar hook is the only Cancel check inside a generate, and a routed update may never reach it.
            comfy.model_management.throw_exception_if_processing_interrupted()
            if total is not None:
                self.total = total
            self.current = value
            run.route(value, self.total)

        def update(self, value: float) -> None:
            self.update_absolute(self.current + value)

    return RoutedProgressBar
