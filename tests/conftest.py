"""Fixtures the unit test modules share."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


class ResidentLoads:
    """Records each CLIP the toolkit's skip_resident_loads scope was entered with, and whether a scope is open now."""

    def __init__(self):
        self.clips = []
        self.open = False

    @contextmanager
    def skip_resident_loads(self, clip):
        self.clips.append(clip)
        self.open = True
        try:
            yield
        finally:
            self.open = False


@pytest.fixture
def resident_loads(monkeypatch):
    """Replaces the toolkit's skip_resident_loads with a recording scope, since the tests' CLIPs have no load_model."""
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    record = ResidentLoads()

    monkeypatch.setattr(toolkit, "skip_resident_loads", record.skip_resident_loads)
    return record


class StripClassifier:
    """Reads "cat" on any image with a bright pixel, "dog" on every image and "owl" only on a square image.

    On the square image of strip_engine, cat sits in the top-left ninth, dog fills every strip and owl no strip.
    """

    def __init__(self):
        self.asked = []

    def nouls(self, statements, *, image=None, state=""):
        self.asked.append((tuple(statements), tuple(image.shape)))
        return [self._answer(statement.rsplit(" ", 1)[-1], image) for statement in statements]

    @staticmethod
    def _answer(tag, image):
        if tag == "cat":
            return 1.0 if float(image.max()) > 0.5 else 0.0
        if tag == "dog":
            return 1.0
        if tag == "owl":
            return 1.0 if image.shape[1] == image.shape[2] else 0.0
        return 0.0


@pytest.fixture
def strip_engine(monkeypatch, resident_loads):
    """Runs the nodes' real locate stage over a StripClassifier, and keeps every proposed tag without a CLIP.

    The toolkit's strip resampling becomes a plain crop, since the tests' comfy.utils has no common_upscale. The
    Tagger's tag_image keeps record.proposed, so its runs reach the locate stage without a generate.
    """
    torch = pytest.importorskip("torch")
    toolkit = pytest.importorskip("logit_classifier.toolkit.comfyui")
    locate = pytest.importorskip("logit_classifier.toolkit.comfyui.locate")
    from logit_tagger import engine, tagging

    classifier = StripClassifier()
    record = SimpleNamespace(classifier=classifier, built=[], proposed=("cat", "dog", "owl"))

    def build_classifier(clip, node, *, needs_images=True):
        record.built.append((clip, node))
        return classifier

    def tag_image(clip, classifier, image, *, prompt_tags=(), on_stage=None, thing_cache, settings=None):
        return tagging.TagTrace(
            size_in=(int(image.shape[1]), int(image.shape[2])), size_out=(int(image.shape[1]), int(image.shape[2])),
            reply="", proposed=record.proposed, candidates=record.proposed, origins=("model",) * len(record.proposed),
            dropped=(), scores=(1.0,) * len(record.proposed), kept=record.proposed, tags=", ".join(record.proposed),
            propose_ms=0.0, verify_ms=0.0,
        )

    @contextmanager
    def shared_vision_encode(clip):
        yield

    def top_left_image():
        image = torch.zeros(1, 9, 9, 3)
        image[:, :3, :3, :] = 1.0
        return image

    monkeypatch.setattr(engine, "build_classifier", build_classifier)
    monkeypatch.setattr(tagging, "tag_image", tag_image)
    monkeypatch.setattr(toolkit, "shared_vision_encode", shared_vision_encode)
    monkeypatch.setattr(locate, "resample_image", lambda image, pixels: image[..., :3])
    record.image = top_left_image
    return record
