"""progress.py: the run's one progress bar, core's token bars routed into it, and the status line.

Fake comfy.utils, comfy.model_management and server modules stand in for ComfyUI, so no comfy install
and no GPU are needed.
"""

import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from logit_tagger import progress

PACK_ROOT = Path(__file__).parent.parent.resolve()
NODE = "Some Node"
NODE_ID = "7"
FINISHED_TWO = r"2 images in \d+\.\d s"


class RecordingProgressBar:
    """Stands in for comfy.utils.ProgressBar and records every (value, total) it is given."""

    instances: ClassVar[list] = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.updates = []
        RecordingProgressBar.instances.append(self)

    def update_absolute(self, value, total=None, preview=None):
        self.updates.append((value, total))


class CancelOnFinalUpdateBar(RecordingProgressBar):
    """Checks for Cancel on its final update, as core's ProgressBar hook does (main.py, hijack_progress)."""

    def update_absolute(self, value, total=None, preview=None):
        super().update_absolute(value, total, preview)
        if value >= (self.total if total is None else total):
            sys.modules["comfy.model_management"].throw_exception_if_processing_interrupted()


class Interrupted(BaseException):
    """Stands in for core's InterruptProcessingException, a BaseException."""


class RecordingServer:
    """Stands in for PromptServer.instance and records every status line it is sent."""

    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def send_progress_text(self, text, node_id):
        if self.fail:
            raise RuntimeError("websocket closed")
        self.sent.append((text, node_id))


@pytest.fixture
def fake_comfy(monkeypatch):
    """Fake comfy.utils, comfy.model_management and server modules, and the records they keep."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    model_management = types.ModuleType("comfy.model_management")
    server = types.ModuleType("server")
    state = SimpleNamespace(interrupt_checks=0, interrupted=False, server=RecordingServer())

    def throw_exception_if_processing_interrupted():
        state.interrupt_checks += 1
        if state.interrupted:
            raise Interrupted()

    RecordingProgressBar.instances = []
    utils.ProgressBar = RecordingProgressBar
    model_management.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted
    model_management.InterruptProcessingException = Interrupted
    server.PromptServer = SimpleNamespace(instance=state.server)
    comfy.utils = utils
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "server", server)
    return state


def run_bar():
    return RecordingProgressBar.instances[0]


def token_bar(max_length):
    """The bar core's clip.generate builds, through the patched attribute exactly as llama.py does."""
    return sys.modules["comfy.utils"].ProgressBar(max_length)


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_no_torch_no_server_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.progress\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'server', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- the one bar ------------------------------------------------------------------------


def test_one_bar_across_three_items_with_a_token_bar_each_never_decreases_and_ends_at_the_total(fake_comfy):
    with progress.RunProgress(total_items=3, unique_id=NODE_ID, noun="image", node=NODE) as run:
        for index in range(3):
            run.item(index, "writing tags")
            bar = token_bar(20)
            for _token in range(20):
                bar.update(1)
            run.item(index, "checking tags")

    updates = run_bar().updates
    values = [value for value, _total in updates]
    assert len(RecordingProgressBar.instances) == 1
    assert values == sorted(values)
    assert all(total >= value for value, total in updates)
    assert updates[-1] == (3 * progress.ITEM_UNITS, 3 * progress.ITEM_UNITS)


def test_the_run_bar_starts_at_zero_over_the_item_total(fake_comfy):
    progress.RunProgress(total_items=4, unique_id=NODE_ID, noun="image", node=NODE)

    assert run_bar().total == 4 * progress.ITEM_UNITS
    assert run_bar().updates == [(0, 4 * progress.ITEM_UNITS)]


def test_a_token_bar_fills_the_generate_share_at_the_fill_ratio_of_its_budget(fake_comfy):
    with progress.RunProgress(total_items=3, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(1, "writing tags")
        bar = token_bar(100)
        bar.update_absolute(13)
        partway = run_bar().updates[-1][0]
        bar.update_absolute(65)
        at_ratio = run_bar().updates[-1][0]
        bar.update_absolute(100)
        at_budget = run_bar().updates[-1][0]

    assert partway == round((1 + progress.GENERATE_SHARE * 13 / 65) * progress.ITEM_UNITS)
    assert at_ratio == round((1 + progress.GENERATE_SHARE) * progress.ITEM_UNITS)
    assert at_budget == at_ratio


def test_a_clean_exit_fills_the_bar_and_a_raised_exit_does_not(fake_comfy):
    with progress.RunProgress(total_items=3, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        token_bar(10).update_absolute(10)
    with pytest.raises(RuntimeError, match="out of memory"), \
            progress.RunProgress(total_items=3, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        token_bar(10).update_absolute(10)
        raise RuntimeError("out of memory")

    clean_bar, raised_bar = RecordingProgressBar.instances
    assert clean_bar.updates[-1] == (3 * progress.ITEM_UNITS, 3 * progress.ITEM_UNITS)
    assert raised_bar.updates[-1] == (round(progress.GENERATE_SHARE * progress.ITEM_UNITS), 3 * progress.ITEM_UNITS)


def test_a_token_bar_that_restarts_inside_an_item_never_moves_the_bar_back(fake_comfy):
    with progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        token_bar(10).update_absolute(6)
        token_bar(10).update_absolute(1)

    values = [value for value, _total in run_bar().updates]
    assert values == sorted(values)


def test_a_routed_bar_before_the_first_item_changes_nothing(fake_comfy):
    with progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE):
        before = list(run_bar().updates)
        bar = token_bar(10)
        bar.update(5)
        bar.update_absolute(10)
        after = list(run_bar().updates)

    assert after == before
    assert len(fake_comfy.server.sent) == 1
    assert re.fullmatch(FINISHED_TWO, fake_comfy.server.sent[0][0])


# --- the patch window -------------------------------------------------------------------


def test_the_patch_routes_bars_inside_the_run_and_is_undone_on_exit(fake_comfy):
    utils = sys.modules["comfy.utils"]

    with progress.RunProgress(total_items=1, unique_id=NODE_ID, noun="image", node=NODE):
        inside = utils.ProgressBar

    assert inside is not RecordingProgressBar
    assert utils.ProgressBar is RecordingProgressBar


def test_the_patch_is_undone_and_the_status_says_failed_after_a_raise(fake_comfy):
    utils = sys.modules["comfy.utils"]

    with pytest.raises(RuntimeError, match="out of memory"), \
            progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        raise RuntimeError("out of memory")

    assert utils.ProgressBar is RecordingProgressBar
    assert fake_comfy.server.sent[-1] == ("failed", NODE_ID)


# --- the status line --------------------------------------------------------------------


def test_the_status_line_for_one_item_is_the_stage_alone_and_sent_only_on_change(fake_comfy):
    with progress.RunProgress(total_items=1, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        run.item(0, "writing tags")
        run.item(0, "checking tags")

    assert fake_comfy.server.sent[:2] == [("writing tags", NODE_ID), ("checking tags", NODE_ID)]
    assert len(fake_comfy.server.sent) == 3
    assert re.fullmatch(r"1 image in \d+\.\d s", fake_comfy.server.sent[2][0])


def test_the_status_line_for_three_items_names_the_noun_and_the_position(fake_comfy):
    with progress.RunProgress(total_items=3, unique_id=NODE_ID, noun="text", node=NODE) as run:
        for index in range(3):
            run.item(index, "checking")

    assert [text for text, _node_id in fake_comfy.server.sent][:3] == [
        "text 1/3: checking",
        "text 2/3: checking",
        "text 3/3: checking",
    ]
    assert len(fake_comfy.server.sent) == 4
    assert re.fullmatch(r"3 texts in \d+\.\d s", fake_comfy.server.sent[3][0])


@pytest.mark.parametrize(("count", "pattern"), [(2, r"^2 images in \d+\.\d s$"), (1, r"^1 image in \d+\.\d s$")])
def test_a_finished_run_ends_on_its_item_count_and_duration(fake_comfy, count, pattern):
    with progress.RunProgress(total_items=count, unique_id=NODE_ID, noun="image", node=NODE) as run:
        for index in range(count):
            run.item(index, "writing tags")

    assert re.match(pattern, fake_comfy.server.sent[-1][0])


def test_a_cancelled_run_ends_on_cancelled_and_the_interrupt_propagates(fake_comfy):
    with pytest.raises(Interrupted), \
            progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        raise Interrupted()

    assert fake_comfy.server.sent[-1] == ("cancelled", NODE_ID)


def test_a_cancel_raised_by_the_clean_exit_fill_ends_on_cancelled_and_propagates(fake_comfy, monkeypatch):
    monkeypatch.setattr(sys.modules["comfy.utils"], "ProgressBar", CancelOnFinalUpdateBar)

    with pytest.raises(Interrupted), \
            progress.RunProgress(total_items=1, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "checking")
        fake_comfy.interrupted = True

    assert fake_comfy.server.sent == [("checking", NODE_ID), ("cancelled", NODE_ID)]


def test_a_failed_run_ends_on_failed_and_the_error_propagates(fake_comfy):
    with pytest.raises(ValueError, match="bad input"), \
            progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        raise ValueError("bad input")

    assert fake_comfy.server.sent[-1] == ("failed", NODE_ID)


def test_item_checks_for_an_interrupt_before_it_announces_the_item(fake_comfy):
    with progress.RunProgress(total_items=2, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")
        fake_comfy.interrupted = True
        with pytest.raises(Interrupted):
            run.item(1, "writing tags")
        after_interrupt = run_bar().updates[-1][0]

    assert fake_comfy.interrupt_checks == 2
    assert fake_comfy.server.sent[0] == ("image 1/2: writing tags", NODE_ID)
    assert len(fake_comfy.server.sent) == 2
    assert re.fullmatch(FINISHED_TWO, fake_comfy.server.sent[1][0])
    assert after_interrupt == 0


def test_a_failed_status_send_never_fails_the_run(fake_comfy):
    fake_comfy.server.fail = True

    with progress.RunProgress(total_items=1, unique_id=NODE_ID, noun="image", node=NODE) as run:
        run.item(0, "writing tags")

    assert run_bar().updates[-1] == (progress.ITEM_UNITS, progress.ITEM_UNITS)


def test_a_run_without_a_node_id_sends_no_status(fake_comfy):
    with progress.RunProgress(total_items=1, unique_id=None, noun="image", node=NODE) as run:
        run.item(0, "writing tags")

    assert fake_comfy.server.sent == []


# --- library loading --------------------------------------------------------------------


def test_a_run_without_the_toolkit_names_the_node_and_the_pip_command_before_any_bar(fake_comfy, monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)

    with pytest.raises(RuntimeError, match=rf'^{NODE}: .*pip install -U "logit-classifier>=0\.6\.0"'):
        progress.RunProgress(total_items=1, unique_id=NODE_ID, noun="image", node=NODE)
    assert RecordingProgressBar.instances == []
