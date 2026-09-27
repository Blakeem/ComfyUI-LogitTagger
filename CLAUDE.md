# ComfyUI Logit Tagger, project guide

A ComfyUI node pack built on the `logit-classifier` library. Every node reads the Qwen3-VL-4B text
encoder the workflow already loads (Krea 2's CLIP) and reads its answers from the logits, so no
second model loads. The Tagger's stages come from the library's toolkit (`logit_classifier.toolkit`),
which ContextAnchoredTileRefine shares. Ideas for later nodes live in `TODO.md`.

## Nodes

| Node | Status | Inputs | Outputs |
|---|---|---|---|
| Logit Tagger | built | `clip`, `image`, optional `prompt`, `instruction` and `tags` sockets | `tags` STRING list, `images` IMAGE list |
| Logit Image Check | built | `clip`, `image`, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_images`, `no_images` |
| Logit Text Check | built | `clip`, `text` socket, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_texts`, `no_texts` |
| Logit Prompt Check | built | `clip`, `image`, `prompt` socket | `all_present`, `score`, `missing`, `images`, `complete_images`, `incomplete_images` |
| Logit Image Classify | built | `clip`, `image`, `question` and `options` widgets | `choice`, `index`, `confidence`, `images` |

A connected `tags` list makes the Tagger check only those tags, and it rejects a prompt or an
instruction connected beside it.

Shared contract for every node:

- V1 contract (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`), `INPUT_IS_LIST = True`, `OUTPUT_NODE = True`,
  `CATEGORY = "Logit Tagger"`, hidden `unique_id`, a tooltip on every input.
- A batch or a list of batches splits into `[1, H, W, C]` images (`lists.split_images`). Per-image
  outputs are lists aligned with those images.
- A text input pairs one to all or one to one (`lists.pair_texts`). A widget under
  `INPUT_IS_LIST` arrives as a list, and differing values are an error (`lists.single_value`).
- One progress bar per run, with core's per-token generate bars routed into it, and a status line
  under the node (`progress.RunProgress`). The run ends on "4 images in 9.3 s", "cancelled" or
  "failed", since frontend 1.53.6 ignores an empty progress text. A `ui.text` display below the
  inputs (`web/logit_tagger.js`).
- An empty yes or no side is `[ExecutionBlocker(None)]`, which skips the nodes wired to it
  (`nodes._split_sides`). Logit Prompt Check splits its images the same way.
- Every class carries `OUTPUT_TOOLTIPS` and `SEARCH_ALIASES`, which core sends to the frontend
  (`server.py:783-796`).
- The logit-only paths (Image Check, Classify, Prompt Check, tag list mode) read images at
  `engine.CHECK_MAX_PIXELS` (0.5 MP). The Tagger's propose and verify read them at
  `engine.VL_MAX_PIXELS` (1 MP), since they share one vision encode.
- Errors name the node and the fix.
- Every tuning value (wording, threshold, budget, cap) is a backend constant set by a script in
  `tests-AB/`. A long text arrives as a socket. A simple question is a widget with a measured default,
  and a blank widget uses that default.

## Modules

```
nodes.py ─┬─> tagging.py ─┬─> engine.py ──> logit_classifier.toolkit.comfyui (classifier) ──> logit_classifier
          │               └─> logit_classifier.toolkit (tags, comfyui tagger and generate)
          ├─> engine.py
          ├─> lists.py
          └─> progress.py ──> logit_classifier.toolkit.comfyui (progress)
```

| Module | Holds |
|---|---|
| `logit_tagger/engine.py` | what every node shares: the pixel caps `VL_MAX_PIXELS` (1 MP) and `CHECK_MAX_PIXELS` (0.5 MP), `check_images`, `library_error` (the pip command), `build_classifier` (the toolkit's `comfy_classifier`), `choose` |
| `logit_tagger/tagging.py` | the Tagger's runs and the prompt check over the toolkit: `tag_settings`, `tag_image`, `parse_tag_list`, `check_tag_list`, `tag_images`, `check_prompt`, `TagTrace`, `PromptCheck` |
| `logit_tagger/lists.py` | list input rules |
| `logit_tagger/progress.py` | `RunProgress`, over the toolkit's `routed_progress_bars`, `send_status` and `run_outcome` |
| `logit_tagger/nodes.py` | every node class |
| `web/logit_tagger.js` | the result display below each node |
| `example_workflows/` | four workflows core lists in its template browser, each run once on this machine |
| `tests-AB/` | measurement harnesses, bootstrapped by `ab_env.py` over the library's `tests-AB/comfy_env.py` |

## The tagger pipeline

The stages are the toolkit's `prompt_tags` and `tag_picture`, driven by `tagging.tag_settings()`,
which returns the toolkit's `TagSettings` defaults. ContextAnchoredTileRefine runs the same stages on
tiles with its own preset values.

```
prompt ──> [language gate] ──> [translate] ──> [prompt stage] ──┐  once per distinct prompt, text only
           noul "Is this text   generate,        list, keep what │
           written in English?" only below 0.5   the English     │
                                                 text grounds    │
                                                 and the thing   │
                                                 check calls a   │
                                                 thing           │
image ──> [resize] ──> [propose] ──> [merge] ──> [thing check] ─┴──> [verify] ──> [clean] ──> tags
          1 MP cap     greedy, stops  model tags   drops a tag            noul per tag   drop_subsets
                       at 25 tags or  first, then  whose p("other")       0.999, or      (words rule),
                       a repeat       prompt tags, reaches 0.9            0.9999 for     score order
                                      64 at most                          prompt-only
```

- A propose reply that holds only the instruction's category nouns gets one more propose with the
  settings' `echo_fallback_instruction`. A connected instruction never triggers it.
- The thing check judges every tag. On the model's own tags it drops some correct ones, such as
  "watermark" and "pixel art", and it removes more vague ones (see Tagger A/B).
- The prompt stage grounds each listed thing in the English text's words, and two ungrounded tags in
  a row end the list. That guard stops the padding the model writes past a prompt's own things, and
  the translation keeps it working for a prompt in another language.

- Propose and verify run inside the toolkit's `shared_vision_encode`, so the vision tower runs once
  per image.
- The toolkit's `generate_text` writes the Qwen chat template itself, since core's `thinking=False`
  appends an empty `<think>` block that made Qwen3-VL-4B-Instruct reply empty. Its `clip_generate`
  clears core's CUDA graph decode queues after each generate (core issue #16441), and `stop_when`
  ends core's decode loop on a predicate over the complete tags so far.
- The CLIP loader types `krea2`, default and `mage` gave identical tags. `flux` and `flux2` load the
  encoder without its vision tower, and `build_classifier` rejects them by name. A text-only Qwen3
  fails the library's model check, and `build_classifier` names the node and the Load CLIP fix.

## Tagger A/B

`tests-AB/ab_tagger_compare.py` tags 72 Krea 2 renders without and with their prompts and 30 COCO
photos, per arm. Blind judges grade each kept tag by `tests-AB/data/tagger-ab/RUBRIC.md` into
`verdicts.json` (102 images, 3,309 tags), and `tests-AB/ab_tagger_score.py` scores the arms.
Precision is yes over yes, no and vague. All arms ran with DynamicVRAM on. Each run's records sit in
`tests-AB/data/tagger-ab/runs/<experiment>/`.

| Condition | Arm | Precision | Correct tags per image | Seconds per image |
|---|---|---|---|---|
| renders, no prompt | the first tagger, `4cbd732` | 0.912 | 16.6 | 6.90 |
| renders, no prompt | the port, `934f901` | 0.940 | 12.8 | 2.76 |
| renders, no prompt | round 2 arm, shipped in `53d9c34` | 0.918 | 15.3 | 2.51 |
| renders, with prompt | the first tagger | 0.900 | 16.4 | 7.66 |
| renders, with prompt | the port | 0.945 | 14.1 | 3.77 |
| renders, with prompt | shipped | 0.923 | 16.7 | 3.50 |
| COCO photos | the first tagger | 0.898 | 10.9 | 4.28 |
| COCO photos | the port | 0.945 | 8.6 | 1.77 |
| COCO photos | shipped | 0.925 | 11.5 | 1.82 |
| renders, no prompt | toolkit defaults, `3bd946c` | 0.946 | 12.8 | 2.23 |
| renders, with prompt | toolkit defaults | 0.950 | 14.0 | 3.36 |
| COCO photos | toolkit defaults | 0.964 | 9.7 | 1.58 |

The owner ranks fewer wrong tags above more correct ones. So the round 2 arm's three changes were
measured again on the fp8 encoder, as a replay of 16 settings from one superset run
(`ab_tagger_score.py --replay`, `runs/2026-09-27-settings`). Each row changes one field from cap 25,
the thing check on every tag, the words rule and verify 0.9:

| Change | Correct per render | Wrong and vague per render |
|---|---|---|
| propose cap 40 over 25 | +0.85 | +0.15 |
| thing check on the prompt's tags only | +0.88 | +0.30 |
| verify threshold 0.999 over 0.9 | -0.21 | -0.08 |

The head-noun subsets rule added no wrong tag on renders. It kept single words beside a longer kept
tag ("bread" beside "loaf of bread"), and on ContextAnchoredTileRefine's tiles it added 0.03 wrong tags
per tile, so the words rule stays. The defaults kept 0.12 wrong and 0.61 vague tags per render against
0.28 and 1.08 for the round 2 arm. A real run of the defaults matched the replay (`runs/2026-09-27-defaults`).

## Measured question shapes

On this model through the ComfyUI backend. Scripts: `tests-AB/ab_question_shapes.py` (60 COCO val2017
photos, built by `tests-AB/build_coco60.py`) and `tests-AB/ab_release_gates.py` (72 Krea 2 renders
labeled by two blind judges, 80 labeled prompts). The render rows come from `ab_release_gates.py` at
the shipped 0.5 MP cap.

| Shape | Result |
|---|---|
| image yes/no, question wording | AUC 0.996 on photos, 0.986 or more for every presence, scene and style question on renders, accuracy 0.970 at 0.5 |
| image yes/no, statement wording | "This image contains an orange" read the color: use question wording for free questions |
| text yes/no, "Does this prompt ask for X in the image?" | accuracy 0.972, says no to negated mentions ("no cats") and decoy words ("dogwood") |
| fixed tag list, "Is there X in this image?" | precision 0.971, recall 0.748 at 0.9999 on COCO |
| per-thing missing list | precision 0.972, recall 0.945 at 0.5 on renders, and a near miss's substituted thing flagged 0.944 |
| image complete, every listed thing at 0.5 or more | accuracy 0.938 on 144 image and prompt pairs, 0.917 on own prompts and 0.958 on near misses |
| whole-prompt yes/no | AUC 0.899 on near-miss prompts only: report per thing instead |
| text yes/no, "Is this text written in English?" | 0.000 to 0.027 on 13 prompts in 10 other languages, 1.000 on 10 English prompts (`tests-AB/ab_prompt_language.py`) |
| choice over 8 styles | accuracy 0.889, confidence AUC 0.947 on renders |
| score | ranks brightness at Spearman 0.99, but "sharp" also falls with darkness. Not shipped |
| ranking written prompts | at chance against a blind judge. Not shipped |

On a 1 MP render read at the 0.5 MP cap, one question costs about 280 ms and 27 packed questions
about 385 ms. At `--max-pixels 1048576` they cost 553 and 650 ms, and every probe at 0.5 MP stayed
inside its 1 MP bootstrap interval. At 0.25 MP style accuracy fell to 0.806, so 0.5 MP ships. On a
COCO photo of about 0.3 MP one question costs about 160 ms.

## Environment

ComfyUI on this machine is the Desktop build, so its python source sits apart from its data.

| Part | Path |
|---|---|
| server source, read this one | `C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI` |
| base path (models, custom_nodes, input, output, `.venv`) | `C:\Users\Blake\Documents\ComfyUI` |
| venv python | `C:\Users\Blake\Documents\ComfyUI\.venv\Scripts\python.exe` |
| library source | `E:\logit-classifier`, which has its own `CLAUDE.md` |
| reference pack for patterns | `C:\Users\Blake\Documents\ComfyUI\custom_nodes\ComfyUI-ContextAnchoredTileRefine` |
| Qwen3-VL-4B text encoders | `models\clip\qwen3vl_4b_fp8_scaled.safetensors`, `models\clip\qwen3-vl-4b-heretic_int8.safetensors` |

The library is installed into the ComfyUI venv as an editable install, so an edit under
`E:\logit-classifier\src` is live on the next ComfyUI start with no sync step.

```
.venv\Scripts\uv.exe pip install -e E:\logit-classifier --python .venv\Scripts\python.exe
```

A Registry install gets the published wheel instead, through `[project] dependencies`.

## Dependency rules

- `[project] dependencies` names the bare `logit-classifier>=0.3.0`, never an extra. The `[hf]` extra
  names torch, and pip resolving torch beside ComfyUI's pinned CUDA build breaks the install.
- Every import of `logit_classifier`, comfy and torch is lazy, inside the function that needs it. A
  missing library raises an error that prints the pip command.
- A bug in the library is fixed in `E:\logit-classifier` and not patched around here. The owner
  publishes the fix to PyPI.
- The library's toolkit holds the generate side of the tagger. Its first prime directive bars
  `generate` only for a classifier answer.

## Verified against the installed core

Read from the server source at commit `88ab4a06` on 2026-09-27.

| Fact | Where |
|---|---|
| Krea 2 loads the entire Qwen3-VL-4B, lm_head mapping included | `comfy/sd.py` |
| `BaseGenerate.generate` builds a `ProgressBar(max_length)` and hands logits to `self.sample_token` | `comfy/text_encoders/llama.py` |
| graph decode runs only when core's dynamic VRAM (`comfy_aimdo`) is on and CUDA graphs are not disabled | `comfy/model_prefetch.py:29`, `llama.py` |
| a text starting with `<\|im_start\|>` bypasses the chat template | `comfy/text_encoders/qwen3vl.py` |
| the image path builds `visual_pos_masks` at batch 1, so one image per request | `qwen3vl.py` |
| core resizes a VL image only above 12.8 MP | `comfy/text_encoders/qwen_vl.py` |
| a list input holding an `ExecutionBlocker` blocks the node that receives it | `execution.py`, `_async_map_node_over_list` |
| If/Else Switch is lazy and skips the branch not taken. Core has no N-way switch | `comfy_extras/nodes_logic.py` |
| Save Image-Text (to Folder) pairs a list of `[1, H, W, C]` images with texts by order | `comfy_extras/nodes_dataset.py` |
| `PromptServer.instance.send_progress_text(text, node_id)` shows text under a node | `server.py` |

## Tests and gates

- Unit gate: `.venv\Scripts\python.exe -m pytest tests -q` from the pack root, no GPU.
- Lint gate: `uvx ruff@0.16.2 check .` reports zero findings, and `node --check web/logit_tagger.js`.
- Harness: `.venv\Scripts\python.exe tests-AB\ab_tagger.py <image>... [--prompt] [--instruction]`
  prints each stage's result and time. Check `nvidia-smi` is idle first.
- `tests-AB/ab_env.py` holds this machine's paths and memory flags and calls the library's
  `tests-AB/comfy_env.py`, which enables core's DynamicVRAM as `main.py` does. So harness decode takes
  the CUDA graph path at about 30 ms per token on the fp8 encoder. A harness that runs several node
  executions calls `ab_env.end_node_execution()` between them.
- A tuning question runs as one superset: `ab_tagger_compare.py --variants V.json --record-thing-scores`
  at the loosest value of each field, then `ab_tagger_score.py --replay ARMS.json --results ...`
  derives the stricter arms offline. `ab_judge.py store` copies a run into `runs/<experiment>/`,
  `ab_judge.py split` writes blind judge batches with calibration pairs, and `ab_judge.py merge` adds
  the verdicts to the store.

## Conventions

- Follow ContextAnchoredTileRefine's patterns: a comfy-free module scope with lazy imports, tooltips
  on every input, errors that name the node and the fix.
- `README.md` is for users and follows the succinct-writing skill.
- A number in a document came from a script in `tests-AB/`, named beside it.
- GPL-3.0, matching the library and the other packs.
- One GPU job at a time on this machine, a 24 GB 3090 Ti shared with ComfyUI.
