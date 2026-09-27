"""engine.py: the model access every node of the pack shares.

Duck typed CLIPs and fake classifiers stand in for the Qwen3-VL text encoder and the library's
Classifier, so no comfy install, no GPU and no model are needed.
"""

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from logit_tagger import engine

PACK_ROOT = Path(__file__).parent.parent.resolve()
NODE = "Some Node"
STOP_ID = 0


def an_image(height=8, width=8, channels=3, batch=1):
    torch = pytest.importorskip("torch")
    return torch.rand(batch, height, width, channels)


class FakeClip:
    """Records every tokenize and generate call, and decodes to canned text."""

    def __init__(self, reply="cat, dog", id_count=3):
        self.reply = reply
        self.id_count = id_count
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return {"qwen3vl_4b": [[(text, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return list(range(self.id_count))

    def decode(self, ids):
        return self.reply


class ScriptedTransformer:
    """Returns one scripted token per sample_token call, then a stop token, as core's sampler does."""

    def __init__(self, script):
        self.script = [*script, STOP_ID]
        self.model = SimpleNamespace(config=SimpleNamespace(stop_tokens=[STOP_ID]))

    def sample_token(self, logits, *args, **kwargs):
        torch = pytest.importorskip("torch")
        return torch.tensor([[self.script.pop(0)]])


class DecodingClip(FakeClip):
    """Runs core's decode loop, at comfy/text_encoders/llama.py:1147-1187, over a word per token id."""

    def __init__(self, words):
        super().__init__()
        self.vocabulary = ["<stop>", *dict.fromkeys(words)]
        self.transformer = ScriptedTransformer([self.vocabulary.index(word) for word in words])
        self.cond_stage_model = SimpleNamespace(clip="qwen3vl_4b",
                                                qwen3vl_4b=SimpleNamespace(transformer=self.transformer))

    def generate(self, tokens, **kwargs):
        ids = []

        self.generate_calls.append((tokens, kwargs))
        for _step in range(kwargs["max_length"]):
            ids.append(int(self.transformer.sample_token(None)[0, 0]))
            if ids[-1] == STOP_ID:
                break
        return ids

    def decode(self, ids):
        return "".join(self.vocabulary[token] for token in ids if token != STOP_ID)


def words_of(text):
    return [part for chunk in text.split(", ") for part in (chunk, ", ")][:-1]


class FakeClassifier:
    """Answers each noul question with the probability its statement maps to."""

    def __init__(self, probabilities=None):
        self.probabilities = probabilities or {}
        self.calls = []

    def classify(self, request, *, image=None):
        answers = {}

        self.calls.append((request, image))
        for qid, question in request.questions.items():
            answers[qid] = SimpleNamespace(noul=self.probabilities.get(question.instructions, 0.0))
        return SimpleNamespace(answers=answers), None


@pytest.fixture
def fake_prefetch(monkeypatch):
    """A comfy.model_prefetch whose cleanup_prefetch_queues records each call."""
    comfy = types.ModuleType("comfy")
    model_prefetch = types.ModuleType("comfy.model_prefetch")
    cleanups = []

    model_prefetch.cleanup_prefetch_queues = lambda: cleanups.append(True)
    comfy.model_prefetch = model_prefetch
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", model_prefetch)
    return cleanups


# --- module scope ---------------------------------------------------------------------


def test_module_scope_imports_no_comfy_no_torch_and_no_library():
    script = (
        "import sys\n"
        "import logit_tagger.engine\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('comfy', 'torch', 'logit_classifier')]\n"
        "print(loaded)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], cwd=PACK_ROOT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


# --- resize_for_vl --------------------------------------------------------------------


def test_resize_keeps_an_image_under_the_cap():
    image = an_image(16, 24)

    resized = engine.resize_for_vl(image)
    assert resized.shape == (1, 16, 24, 3)
    assert resized.equal(image)


def test_resize_caps_pixels_and_keeps_aspect(monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 100)
    image = an_image(40, 20)

    resized = engine.resize_for_vl(image)
    height, width = resized.shape[1], resized.shape[2]
    assert height * width <= 100
    assert (height, width) == (14, 7)


def test_resize_never_upscales(monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 10_000)
    image = an_image(4, 5)

    assert engine.resize_for_vl(image).shape == (1, 4, 5, 3)


def test_resize_keeps_only_three_channels(monkeypatch):
    monkeypatch.setattr(engine, "VL_MAX_PIXELS", 16)
    small = an_image(4, 4, channels=4)
    large = an_image(8, 8, channels=4)

    assert engine.resize_for_vl(small).shape == (1, 4, 4, 3)
    assert engine.resize_for_vl(large).shape == (1, 4, 4, 3)


# --- check_images ---------------------------------------------------------------------


def test_check_images_accepts_rgb_and_rgba_batches():
    engine.check_images(an_image(batch=2), NODE)
    engine.check_images(an_image(channels=4), NODE)


def test_check_images_rejects_an_image_that_is_not_4d_naming_the_node():
    with pytest.raises(ValueError, match=f"^{NODE}: .*\\[B, H, W, C\\]"):
        engine.check_images(an_image()[0], NODE)


def test_check_images_rejects_fewer_than_3_channels_naming_the_node():
    with pytest.raises(ValueError, match=f"^{NODE}: .*RGB or RGBA"):
        engine.check_images(an_image(channels=1), NODE)


def test_check_images_rejects_an_object_without_a_shape():
    with pytest.raises(ValueError, match=NODE):
        engine.check_images(object(), NODE)


# --- build_classifier -----------------------------------------------------------------


def test_missing_library_names_the_node_the_version_and_the_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier", None)

    with pytest.raises(RuntimeError) as raised:
        engine.build_classifier(FakeClip(), NODE)
    message = str(raised.value)
    assert message.startswith(f"{NODE}: ")
    assert "logit-classifier>=0.3.0" in message
    assert "pip install -U logit-classifier" in message


def test_library_without_the_comfy_backend_gets_the_same_message(monkeypatch):
    pytest.importorskip("logit_classifier")
    monkeypatch.setitem(sys.modules, "logit_classifier.backends.comfy_clip", None)

    with pytest.raises(RuntimeError, match=r"pip install -U logit-classifier"):
        engine.build_classifier(FakeClip(), NODE)


def test_library_older_than_0_3_0_gets_the_same_message(monkeypatch):
    comfy_clip = pytest.importorskip("logit_classifier.backends.comfy_clip")
    monkeypatch.delattr(comfy_clip, "shared_vision_encode")

    with pytest.raises(RuntimeError, match=r"logit-classifier>=0\.3\.0"):
        engine.build_classifier(FakeClip(), NODE)


@pytest.fixture
def recording_classifier(monkeypatch):
    """Replaces the library's backend and Classifier, and records what the Classifier was built with."""
    comfy_clip = pytest.importorskip("logit_classifier.backends.comfy_clip")
    import logit_classifier

    built = {}

    class Backend:
        sees_images = True

        def __init__(self, clip):
            self.clip = clip

    class RecordingClassifier:
        def __init__(self, config, backend):
            built["config"] = config
            built["backend"] = backend

    monkeypatch.setattr(comfy_clip, "ComfyClipBackend", Backend)
    monkeypatch.setattr(logit_classifier, "Classifier", RecordingClassifier)
    built["backend_class"] = Backend
    return built


def test_build_classifier_turns_off_the_prior_and_the_calibration_file(recording_classifier):
    clip = FakeClip()

    engine.build_classifier(clip, NODE)
    assert recording_classifier["config"].use_prior_debias is False
    assert recording_classifier["config"].calibration_path is None
    assert recording_classifier["backend"].clip is clip


def test_a_clip_without_a_vision_tower_is_rejected_naming_the_node(recording_classifier):
    recording_classifier["backend_class"].sees_images = False

    with pytest.raises(ValueError, match=f"^{NODE}: .*vision tower.*krea2"):
        engine.build_classifier(FakeClip(), NODE)
    assert "config" not in recording_classifier


def test_a_text_only_node_accepts_a_clip_without_a_vision_tower(recording_classifier):
    recording_classifier["backend_class"].sees_images = False

    engine.build_classifier(FakeClip(), NODE, needs_images=False)
    assert recording_classifier["config"].use_prior_debias is False


# --- clip_generate --------------------------------------------------------------------


def test_clip_generate_returns_the_ids_and_cleans_up(fake_prefetch):
    clip = FakeClip(id_count=4)

    ids = engine.clip_generate(clip, "tokens", do_sample=False, max_length=9)
    assert ids == [0, 1, 2, 3]
    assert clip.generate_calls == [("tokens", {"do_sample": False, "max_length": 9})]
    assert fake_prefetch == [True]


def test_clip_generate_cleans_up_after_a_raise(fake_prefetch):
    class FailingClip(FakeClip):
        def generate(self, tokens, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        engine.clip_generate(FailingClip(), "tokens")
    assert fake_prefetch == [True]


def test_clip_generate_skips_cleanup_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", None)

    assert engine.clip_generate(FakeClip(id_count=2), "tokens") == [0, 1]


def test_clip_generate_skips_cleanup_when_core_lacks_it(fake_prefetch, monkeypatch):
    monkeypatch.delattr(sys.modules["comfy.model_prefetch"], "cleanup_prefetch_queues")

    assert engine.clip_generate(FakeClip(id_count=2), "tokens") == [0, 1]
    assert fake_prefetch == []


# --- stop_when ------------------------------------------------------------------------


def test_stop_when_ends_the_decode_once_the_predicate_holds():
    clip = DecodingClip(words_of("cat, dog, fox, owl"))
    seen = []

    def two_tags(tags):
        seen.append(list(tags))
        return len(tags) >= 2

    with engine.stop_when(clip, two_tags):
        ids = clip.generate(None, max_length=50)
    assert clip.decode(ids) == "cat, dog"
    assert ["cat"] in seen
    assert seen[-1] == ["cat", "dog"]


def test_stop_when_restores_an_unshadowed_sample_token_after_success_and_a_raise():
    clip = DecodingClip(words_of("cat, dog"))

    with engine.stop_when(clip, lambda tags: False):
        clip.generate(None, max_length=50)
    assert "sample_token" not in vars(clip.transformer)
    with pytest.raises(RuntimeError, match="boom"), engine.stop_when(clip, lambda tags: False):
        raise RuntimeError("boom")
    assert "sample_token" not in vars(clip.transformer)


def test_stop_when_restores_a_shadowed_sample_token_after_success_and_a_raise():
    clip = DecodingClip(words_of("cat, dog"))
    shadow = clip.transformer.sample_token
    clip.transformer.sample_token = shadow

    with engine.stop_when(clip, lambda tags: False):
        assert vars(clip.transformer)["sample_token"] is not shadow
    assert vars(clip.transformer)["sample_token"] is shadow
    with pytest.raises(RuntimeError, match="boom"), engine.stop_when(clip, lambda tags: False):
        raise RuntimeError("boom")
    assert vars(clip.transformer)["sample_token"] is shadow


def test_stop_when_is_a_no_op_for_a_clip_without_core_sampler():
    ran = []

    with engine.stop_when(FakeClip(), lambda tags: True):
        ran.append(True)
    assert ran == [True]


# --- chat -----------------------------------------------------------------------------


def test_chat_without_an_image():
    assert engine.chat("Hello") == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"


def test_chat_with_an_image_puts_the_image_slot_before_the_text():
    expected = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    assert engine.chat("Hello", image=True) == expected


# --- generate_text --------------------------------------------------------------------


def test_generate_text_sends_the_chat_text_and_the_image_greedy():
    clip = FakeClip(reply="fox, snow")
    image = an_image()

    text = engine.generate_text(clip, "Tag it.", image=image, max_tokens=16)
    (rendered, kwargs), = clip.tokenize_calls
    (_tokens, generate_kwargs), = clip.generate_calls
    assert text == "fox, snow"
    assert rendered == engine.chat("Tag it.", image=True)
    assert kwargs.keys() == {"images"}
    assert kwargs["images"][0] is image
    assert len(kwargs["images"]) == 1
    assert generate_kwargs == {"do_sample": False, "max_length": 16}


def test_generate_text_without_an_image_tokenizes_the_text_alone():
    clip = FakeClip(reply="a red fox")

    assert engine.generate_text(clip, "Write.", max_tokens=16) == "a red fox"
    assert clip.tokenize_calls == [(engine.chat("Write."), {})]


def test_generate_text_drops_the_unfinished_tag_when_the_budget_is_reached():
    clip = FakeClip(reply="fox, snow, red fire hyd", id_count=16)

    assert engine.generate_text(clip, "Tag it.", image=an_image(), max_tokens=16) == "fox, snow"


def test_generate_text_keeps_the_whole_reply_below_the_budget():
    clip = FakeClip(reply="fox, snow, red fire hyd", id_count=15)

    assert engine.generate_text(clip, "Tag it.", image=an_image(), max_tokens=16) == "fox, snow, red fire hyd"


def test_generate_text_stops_on_should_stop_and_restores_the_sampler():
    clip = DecodingClip(words_of("cat, dog, fox, owl"))

    text = engine.generate_text(clip, "Tag it.", image=an_image(), max_tokens=50,
                                should_stop=lambda tags: "dog" in tags)
    assert text == "cat, dog"
    assert "sample_token" not in vars(clip.transformer)


def test_generate_text_goes_through_the_graph_cleanup(fake_prefetch):
    engine.generate_text(FakeClip(), "Tag it.", max_tokens=16)

    assert fake_prefetch == [True]


# --- nouls ----------------------------------------------------------------------------


def test_nouls_sends_one_request_with_one_noul_per_statement_in_order():
    classifier = FakeClassifier({"The sky is blue": 0.9, "The sky is green": 0.1})
    image = an_image()

    probabilities = engine.nouls(classifier, ["The sky is blue", "The sky is green"], image=image)
    (request, sent_image), = classifier.calls
    questions = list(request.questions.values())
    assert probabilities == [0.9, 0.1]
    assert sent_image is image
    assert request.state == ""
    assert [q.type for q in questions] == ["noul", "noul"]
    assert [q.instructions for q in questions] == ["The sky is blue", "The sky is green"]


def test_nouls_passes_a_text_state_without_an_image():
    classifier = FakeClassifier({"It is about cats": 0.7})

    assert engine.nouls(classifier, ["It is about cats"], state="A cat sat.") == [0.7]
    (request, sent_image), = classifier.calls
    assert request.state == "A cat sat."
    assert sent_image is None


def test_nouls_sends_no_request_without_statements():
    classifier = FakeClassifier()

    assert engine.nouls(classifier, [], image=an_image()) == []
    assert classifier.calls == []
