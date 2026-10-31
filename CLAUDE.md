# ComfyUI Logit Tagger, project guide

A ComfyUI node pack built on the `logit-classifier` library. Every node reads the Qwen3-VL-4B text
encoder the workflow already loads (Krea 2's CLIP) and reads its answers from the logits, so no
second model loads. Logit Style Extractor's image path is the one exception, since it writes its style
line as generated text. The Tagger's stages come from the library's toolkit (`logit_classifier.toolkit`),
which ContextAnchoredTileRefine shares. Ideas for later nodes live in the owner's local `TODO.md`, which git
ignores.

## Nodes

| Node | Status | Inputs | Outputs |
|---|---|---|---|
| Logit Tagger | built | `clip`, `image`, optional `prompt` socket and `strict` and `positions` widgets | `tags` STRING list, `images` IMAGE list |
| Logit Custom Tagger | built | `clip`, `image`, `instruction` widget, optional `strict`, `things_only` and `positions` widgets | `tags` STRING list, `images` IMAGE list |
| Logit Tag Check | built | `clip`, `image`, `tags` socket, optional `strict` and `positions` widgets | `tags` STRING list, `images` IMAGE list |
| Logit Image Check | built | `clip`, `image`, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_images`, `no_images` |
| Logit Text Check | built | `clip`, `text` socket, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_texts`, `no_texts` |
| Logit Prompt Check | built | `clip`, `image`, `prompt` socket | `all_present`, `score`, `missing`, `images`, `complete_images`, `incomplete_images` |
| Logit Image Classify | built | `clip`, `image`, `question` and `options` widgets | `choice`, `index`, `confidence`, `images` |
| Logit Text Classify | built | `clip`, `text` socket, `question` and `options` widgets | `choice`, `index`, `confidence`, `texts` |
| Logit Style Extractor | built | `clip`, optional `image` and `prompt` sockets | `style` STRING list |

Logit Tag Check checks only the connected tags and writes none of its own. One `tags` text checks every
image, and one text per image pairs one to one.

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
  (`server.py:869-882`).
- The logit-only paths (Image Check, Image Classify, Prompt Check, Tag Check) read images at
  `engine.CHECK_MAX_PIXELS` (0.5 MP). The Tagger's propose and verify read them at
  `engine.VL_MAX_PIXELS` (1 MP), since they share one vision encode.
- Errors name the node and the fix.
- Logit Style Extractor's image path is the pack's one output the classifier does not check.
- Every tuning value (wording, threshold, budget, cap) is a backend constant set by a script in
  `tests-AB/`. A long text arrives as a socket, except a Custom node's instruction. That instruction is a
  multiline widget holding the measured default, so the user edits the wording in place. A simple
  question is a widget with a measured default, and a blank widget uses that default.

## Modules

```
nodes.py ─┬─> tagging.py ─┬─> engine.py ──> logit_classifier.toolkit.comfyui (classifier) ──> logit_classifier
          │               └─> logit_classifier.toolkit (tags, comfyui tagger and generate)
          ├─> style.py ───┬─> tagging.py, engine.py
          │               └─> logit_classifier.toolkit.comfyui (style)
          ├─> engine.py
          ├─> lists.py
          └─> progress.py ──> logit_classifier.toolkit.comfyui (progress)
```

| Module | Holds |
|---|---|
| `logit_tagger/engine.py` | what every node shares: the pixel caps `VL_MAX_PIXELS` (1 MP) and `CHECK_MAX_PIXELS` (0.5 MP), `check_images`, `LIBRARY_VERSION` (0.6.0), `library_error` (the pip command), `toolkit` (the version gate), `build_classifier` (the toolkit's `comfy_classifier`), `choose` |
| `logit_tagger/tagging.py` | the Tagger's runs, the tag check and the prompt check over the toolkit: `tag_settings`, `custom_settings`, `tag_image`, `parse_tag_list`, `check_tag_list`, `check_tag_lists`, `tag_images`, `check_prompt`, `TagTrace`, `PromptCheck` |
| `logit_tagger/style.py` | Logit Style Extractor's run over the toolkit's `prompt_style` and `style_line`: `STYLE_EXTRACTOR_NAME`, `extract_styles`, `StyleTrace`. A prompt's style words come first, and the entire image's style line is the fallback |
| `logit_tagger/lists.py` | list input rules |
| `logit_tagger/progress.py` | `RunProgress`, over the toolkit's `routed_progress_bars`, `send_status` and `run_outcome` |
| `logit_tagger/nodes.py` | every node class |
| `web/logit_tagger.js` | the result display below each node |
| `workflows/` | eleven workflows core lists in its template browser, each run once on this machine through the frontend on 2026-10-17 |
| `tests-AB/` | measurement harnesses, bootstrapped by `ab_env.py` over the library's `tests-AB/comfy_env.py` |

## The tagger pipeline

The stages are the toolkit's `tag_prompt` and `tag_image`, driven by `tagging.tag_settings(strict)`,
which returns the toolkit's `TagSettings` defaults, or those defaults with `verify_thresholds` set to the
toolkit's `STRICT_THRESHOLDS` when the node's `strict` is on. ContextAnchoredTileRefine runs the same stages on
tiles with its own preset values.

```
prompt ──> [language gate] ──> [translate] ──> [prompt stage] ──┐  once per distinct prompt, text only
           noul "Is this text   generate,        list, keep what │
           written in English?" only below 0.5   the English     │
                                                 text grounds    │
                                                 and the thing   │
                                                 check calls a   │
                                                 thing           │
image ──> [resize] ──> [propose] ──> [merge] ──> [thing check] ─┴──> [verify] ──> [clean] ──> [locate] ──> tags
          1 MP cap     greedy, stops  model tags   drops a tag            noul per tag   drop_subsets  positions on
                       at 25 tags or  first, then  whose p("thing")       0.999, or      (words rule), only, noul per
                       a repeat       prompt tags, is below 0.1           0.9999 for     score order   tag on 6 strips
                                      64 at most                          prompt-only                  at 0.25 MP,
                                                                                                       0.9 to place
```

- The verify box shows the broad thresholds. With `strict` on, every tag needs 0.99998.
- A propose reply that holds only the instruction's category nouns gets one more propose with the
  settings' `echo_fallback_instruction`. Logit Custom Tagger keeps it and the `category_nouns` drop only
  when its instruction equals the default wording.
- The thing check judges every tag. On the model's own tags it drops some correct ones, such as
  "watermark" and "pixel art", and it removes more vague ones (see Tagger A/B). Logit Custom Tagger with
  `things_only` off runs no thing check.
- The prompt stage grounds each listed thing in the English text's words, and two ungrounded tags in
  a row end the list. That guard stops the padding the model writes past a prompt's own things, and
  the translation keeps it working for a prompt in another language.
- The locate stage runs only with a node's `positions` on, in Logit Tagger, Logit Custom Tagger and Logit Tag Check,
  which runs it after its check. It is the toolkit's `locate_tags` on the node's input image at full resolution, with
  the settings' verify statement. A tag no strip holds at 0.9 is dropped, and an axis where one strip alone holds a tag
  names its place, such as top-left. On ContextAnchoredTileRefine's tiles, strips at 0.25 MP gave 338 judged position
  terms and strips at 0.5 MP 303, with 3 wrong in each (its `tests-AB/ab_tags_fp8_arms.py`,
  `tests-AB/tags-bench-log.md` section 12). The threshold 0.9 is ContextAnchoredTileRefine's
  `SHIPPED_TAGS_POSITION_THRESHOLD`, set from one 8K tile (its `context_anchored_tile_refine/captions.py`), and it is
  unmeasured on entire images.

- Propose and verify run inside the toolkit's `shared_vision_encode`, so the vision tower runs once
  per image.
- The toolkit's `generate_text` writes the Qwen chat template itself as one user turn, and a text that starts
  with `<|im_start|>` skips core's template. At `ce1e922d` core's Qwen3-VL tokenizer adds no `<think>` block
  (`comfy/text_encoders/qwen3vl.py`). The Krea 2 CLIP's tokenizer formats a text with no image in
  `KREA2_TEMPLATE`, which adds Qwen-Image's system turn that asks to describe an image's objects
  (`comfy/text_encoders/krea2.py`). So core's template would put that system turn before a text-only request. At
  `88ab4a06` core's `thinking=False` appended an empty `<think>` block, which made Qwen3-VL-4B-Instruct reply empty.
- The toolkit's `clip_generate` clears core's CUDA graph decode queues after each generate (core issue #16441), and
  `stop_when` ends core's decode loop on a predicate over the complete tags so far.
- The CLIP loader types `krea2`, default and `mage` gave identical tags. `flux` and `flux2` load the
  encoder without its vision tower, and `build_classifier` rejects them by name. A text-only Qwen3
  fails the library's model check, and `build_classifier` names the node and the Load CLIP fix.

## Workflows

`workflows/Upscale and fix a face with SD 1.5.json` rests on these facts, which its JSON does not show.

- Core's `SAM3_Detect` returns `bboxes` as one list of `{x, y, width, height, score}` dicts per image, and core's
  `ImageCropV2` reads one dict from its `ImageCrop` widget. So `SAM3_Detect`'s `bboxes` cannot feed `ImageCropV2`. The
  face size check measures the `InpaintCropImproved` crop "Face box" at context 1.0 with no resize instead, through
  `GetImageSize` and `ComfyMathExpression` `max(a, b) <= 256`.
- `SAM3_Detect` and both `InpaintCropImproved` nodes read Logit Image Check's `yes_images`. So a no answer blocks
  `SAM3_Detect`, both crops and Logit Custom Tagger, and the switch "Face found?" never asks for its `on_true` input.
  The SAM3 checkpoint loader and the "Find: face" text encode still run, since `SAM3_Detect` reads them and output
  node Logit Custom Tagger needs `SAM3_Detect`.
- Core runs every output node (`execution.py`), and every pack node is one. So a lazy switch alone would not skip
  Logit Custom Tagger or the SAM3 and crop nodes that feed it.
- Two `ComfySwitchNode`s pick the image to save. "Face found?" reads Logit Image Check's `answer`, and "Small face?"
  at its `on_true` input reads the size check.
- The face prompt comes from Logit Custom Tagger with `things_only` off, since Logit Tagger's tags of the face crop
  left out "face". Its instruction names no headwear, since a wording that named headwear made the model write
  "no headwear".
- On a back view, SAM3 still returned a box small enough to pass the size check. So the Image Check gate is what
  keeps a face refine off an image that shows no face.

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
tag ("bread" beside "loaf of bread"), and on ContextAnchoredTileRefine's tiles it added 0.03 wrong or vague tags
per tile, so the words rule stays. The defaults kept 0.12 wrong and 0.61 vague tags per render against
0.28 and 1.08 for the round 2 arm. A real run of the defaults matched the replay (`runs/2026-09-27-defaults`).
The strict thresholds, 0.99998 for every tag, kept 8.4 correct, 0.01 wrong and 0.33 vague tags per render
(`replay_arms_strict.json`, `replay_strict.txt`).

## Style Extractor A/B

`tests-AB/ab_style.py` runs each path through the toolkit function the node calls. Two blind judges grade every
phrase and line by `tests-AB/data/style/RUBRIC.md`, and each run's records, judge files and `labels.json` sit in
`tests-AB/data/style/<run>/`.

The prompt path's defaults come from one superset, run `2026-10-17-prompt-superset` with
`data/style/variants-prompt.json`: 2 listings by 4 scorings over the 72 render prompts and the 80 textset prompts, with
each threshold replayed offline. The judges agreed on 695 of 697 phrase grades. "Medium named" counts render prompts
whose kept phrases hold every word of the manifest medium, so it misses a medium in other words ("graphite pencil
drawing"). "Wrong" counts phrases graded no, per prompt over both sets.

| Arm | Medium named | Correct per render prompt | Wrong per prompt |
|---|---|---|---|
| first defaults: no none answer, the thing check's question, 0.5 | 0.681 | 1.57 | 0.224 |
| none answer, the made-or-shows question, 0.2 (shipped) | 0.847 | 1.81 | 0.053 |
| the same at 0.7 | 0.750 | 1.68 | 0.013 |
| the same at 0.9 | 0.722 | 1.62 | 0.000 |

The none answer cut textset prompts with a kept phrase from 23 to 8 of 80 under the first scoring at 0.5. The shipped arm's 8
wrong phrases are style neighbors ("8k", "limited color palette", "cinematic lighting", "minimalist"), and none names
an object. 0.7 drops "comic book panel" (p 0.45) on 4 renders, and "urban photograph" and "macro photography". A real
run of the defaults matched the replay on all 152 prompts (`2026-10-17-prompt-defaults`).

A medium inside a subject or shot phrase scores low, such as "night street photograph" at p 0.01 and "interior
photograph" at 0.04, so the prompt path keeps it only when the prompt also names the medium alone. With an image
connected, the node then falls back to the image's style line. A listing that writes the medium first, without shot,
subject or place words, named 60 of 72 against 61 and listed more things, so it does not ship
(`2026-10-17-prompt-medium-first`).

The image path is ContextAnchoredTileRefine's style caption, unchanged (`2026-10-17-image`). Its line named the
render's medium on 49 of 72 renders and named content, such as "a humanoid robot" or "neon-lit urban streets", on 28.
On 30 COCO photos it named the photograph medium on all 30 and content on 5. The judges agreed on 178 of 204 line
grades. A line took 1.55 s on a 1 MP render and 0.91 s on a COCO photo.

## Measured question shapes

On this model through the ComfyUI backend. Scripts: `tests-AB/ab_question_shapes.py` (60 COCO val2017
photos, built by `tests-AB/build_coco60.py`) and `tests-AB/ab_release_gates.py` (72 Krea 2 renders
labeled by two blind judges, 80 labeled prompts). The render rows come from `ab_release_gates.py` at
the shipped 0.5 MP cap.

| Shape | Result |
|---|---|
| image yes/no, question wording | AUC 0.996 on photos, 0.986 or more for every presence, scene and style question on renders, accuracy 0.970 at 0.5 |
| image yes/no, statement wording | "This image contains an orange" read the color. Use question wording for free questions |
| text yes/no, "Does this prompt ask for X in the image?" | accuracy 0.972, says no to negated mentions ("no cats") and decoy words ("dogwood") |
| text yes/no, the person wording | on 112 prompts "a person or people" read accuracy 0.964 and "a person" 0.848, which said yes to 3 of 12 prompts of a group (`tests-AB/ab_person_wording.py`) |
| image yes/no, the person wording | five wordings read accuracy 0.976 to 0.992 on 124 renders and photos, a group included, so "Is there a person in this image?" stays (`tests-AB/ab_person_wording.py`) |
| fixed tag list, "This image visibly contains X" | precision 0.903, recall 0.756 at 0.9999 and 0.950, 0.422 at 0.99998 on COCO. 8 of the 11 wrong tags at 0.9999 were "orange" read as the color |
| presence wording | on 905 judged tile candidates "This image visibly contains X" matched 0.3.0's "Is there a X in this image?" within every paired interval at pooled precision 0.95 and 0.98, and it needs no grammar. On COCO tag lists the 0.3.0 question kept 3.79 correct tags per image at precision 0.98 against 1.90 (`tests-AB/ab_presence_wording.py`, round 4) |
| per-thing missing list | precision 0.976, recall 0.931 at 0.01 on renders, and a near miss's substituted thing flagged 0.931 |
| image complete, every listed thing at 0.01 or more | accuracy 0.938 on 144 image and prompt pairs, 0.931 on own prompts and 0.944 on near misses |
| whole-prompt yes/no | AUC 0.896 on near-miss prompts only. Report per thing instead |
| text yes/no, "Is this text written in English?" | 0.000 to 0.027 on 13 prompts in 10 other languages, 1.000 on 10 English prompts (`tests-AB/ab_prompt_language.py`) |
| choice over 8 styles | Logit Image Classify's default question and option names read accuracy 0.917, confidence AUC 0.934 on renders |
| text choice over 8 styles | three wordings picked 72 of 72 render prompts' media. With each prompt's leading medium phrase removed, "What is the art style of the image this prompt describes?" picked 51 of 72, a paired 0.069 [0.014, 0.125] above "What art style does this prompt ask for?" (`tests-AB/ab_text_classify.py`, run `2026-10-17-cues`) |
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

- `[project] dependencies` names the bare `logit-classifier>=0.6.0`, never an extra. The `[hf]` extra
  names torch, and pip resolving torch beside ComfyUI's pinned CUDA build breaks the install.
- Every node's run calls `engine.toolkit(node)` before its first toolkit call. `engine.toolkit` reads
  `logit_classifier.__version__`, since an editable install can keep a stale version in its dist metadata,
  and rejects a library older than `LIBRARY_VERSION`.
- Every import of `logit_classifier`, comfy and torch is lazy, inside the function that needs it. A
  missing library raises an error that prints the pip command.
- A bug in the library is fixed in `E:\logit-classifier` and not patched around here. The owner
  publishes the fix to PyPI.
- The library's toolkit holds the generate side of the tagger. Its first prime directive bars
  `generate` only for a classifier answer.

## Verified against the installed core

Read from the server source at commit `ce1e922d` on 2026-10-17.

| Fact | Where |
|---|---|
| Krea 2 loads the entire Qwen3-VL-4B, lm_head mapping included | `comfy/sd.py` |
| `BaseGenerate.generate` builds a `ProgressBar(max_length)` and hands logits to `self.sample_token` | `comfy/text_encoders/llama.py` |
| graph decode runs only when core's dynamic VRAM (`comfy_aimdo`) is on and CUDA graphs are not disabled | `comfy/model_prefetch.py:29`, `llama.py` |
| a text starting with `<\|im_start\|>` bypasses the chat template | `comfy/text_encoders/qwen3vl.py` |
| the image path builds `visual_pos_masks` at batch 1, so one image per request | `qwen3vl.py` |
| core resizes a VL image only above 12.8 MP | `comfy/text_encoders/qwen_vl.py` |
| a list input holding an `ExecutionBlocker` blocks the node that receives it | `execution.py`, `_async_map_node_over_list` |
| a node class with `OUTPUT_NODE = True` is an output of the prompt unless the request names `partial_execution_targets`, and core queues every output with the inputs it reads | `execution.py`, `validate_prompt` and `PromptExecutor.execute_async` |
| If/Else Switch is lazy and skips the branch not taken. Core has no N-way switch | `comfy_extras/nodes_logic.py` |
| Save Image-Text (to Folder) pairs a list of `[1, H, W, C]` images with texts by order | `comfy_extras/nodes_dataset.py` |
| `PromptServer.instance.send_progress_text(text, node_id)` shows text under a node | `server.py` |
| the template browser reads a pack's `example_workflows`, `example`, `examples`, `workflow` or `workflows` folder | `app/custom_node_manager.py` |
| `SAM3_Detect` returns `masks`, one mask per image that joins every found object unless `individual_masks` is on, and `bboxes`, one list of `{x, y, width, height, score}` dicts per image | `comfy_extras/nodes_sam3.py` |
| `ImageCropV2` reads its `crop_region` as one dict through `crop_region.get("x")`, `"y"`, `"width"` and `"height"`, set by its `ImageCrop` widget | `comfy_extras/nodes_images.py` |
| `ComfyMathExpression` takes an autogrow `values` input named `a` to `z`, which a saved workflow writes as `values.a` and `values.b`, evaluates `expression` with `max` among its functions, and returns FLOAT, INT and a BOOLEAN output named BOOL | `comfy_extras/nodes_math.py` |
| `SplitImageToTileList` returns a list of `tile_width` by `tile_height` tiles at a stride of the tile size minus `overlap`, and a quarter tile at least. A last tile that would cross the image edge shifts back to end at the edge, so it overlaps the tile before it by more than `overlap` | `comfy_extras/nodes_images.py` |
| `ImageMergeTileList` rebuilds that grid from the first tile's size and its `final_width`, `final_height` and `overlap`. With `overlap` above 0 it weights each tile by a sine window that falls near 0 at the tile's edges, and it divides the summed tiles by the summed weights | `comfy_extras/nodes_images.py` |

## Tests and gates

- Unit gate: `.venv\Scripts\python.exe -m pytest tests -q` from the pack root, no GPU.
- Lint gate: `uvx ruff@0.16.2 check .` reports zero findings, and `node --check web/logit_tagger.js`.
- Harness: `.venv\Scripts\python.exe tests-AB\ab_tagger.py <image>... [--prompt TEXT] [--instruction TEXT] [--keep-non-things] [--positions]`
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
