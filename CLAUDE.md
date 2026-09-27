# ComfyUI Logit Tagger, project guide

A ComfyUI node pack built on the `logit-classifier` library. Every node reads the Qwen3-VL-4B text
encoder the workflow already loads (Krea 2's CLIP) and reads its answers from the logits, so no
second model loads and no generated text is parsed. Ideas for later nodes live in `TODO.md`.

## Nodes

| Node | Status | Inputs | Outputs |
|---|---|---|---|
| Logit Tagger | built | `clip`, `image`, optional `prompt` and `instruction` sockets | `tags` STRING list, `images` IMAGE list |
| Logit Image Check | built | `clip`, `image`, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_images`, `no_images` |
| Logit Text Check | built | `clip`, `text` socket, `question` widget | `answer` BOOLEAN, `probability` FLOAT list, `yes_texts`, `no_texts` |
| Logit Tagger tag list | planned | optional `tags` socket on Logit Tagger | the listed tags the image shows |
| Logit Prompt Check | planned | `clip`, `image`, `prompt` socket | `all_present`, `score`, `missing`, `images` |
| Logit Image Classify | planned | `clip`, `image`, `question` and `options` widgets | `choice`, `index`, `confidence` |

Shared contract for every node:

- V1 contract (`INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`), `INPUT_IS_LIST = True`, `OUTPUT_NODE = True`,
  `CATEGORY = "Logit Tagger"`, hidden `unique_id`, a tooltip on every input.
- A batch or a list of batches splits into `[1, H, W, C]` images (`lists.split_images`). Per-image
  outputs are lists aligned with those images.
- A text input pairs one to all or one to one (`lists.pair_texts`). A widget under
  `INPUT_IS_LIST` arrives as a list, and differing values are an error (`lists.single_value`).
- One progress bar per run, with core's per-token generate bars routed into it, and a status line
  under the node (`progress.RunProgress`). A `ui.text` display below the inputs
  (`web/logit_tagger.js`).
- An empty yes or no side is `[ExecutionBlocker(None)]`, which skips the nodes wired to it.
- Errors name the node and the fix.
- Every tuning value (wording, threshold, budget, cap) is a backend constant set by a script in
  `tests-AB/`. A long text arrives as a socket. A simple question is a widget with a measured default,
  and a blank widget uses that default.

## Modules

```
nodes.py ──> tagging.py ──> engine.py ──> logit_classifier (Classifier, ComfyClipBackend, tags)
   │                           ^
   ├──> lists.py               │
   ├──> progress.py            │
   └───────────────────────────┘
```

| Module | Holds |
|---|---|
| `logit_tagger/engine.py` | model access every node shares: `resize_for_vl` (1 MP cap), `build_classifier`, `clip_generate` (CUDA graph cleanup), `stop_when`, `chat`, `generate_text`, `nouls` |
| `logit_tagger/tagging.py` | the tagger pipeline: propose, prompt things, merge, thing check, verify, clean. `tag_image`, `tag_images`, `TagTrace` |
| `logit_tagger/lists.py` | list input rules |
| `logit_tagger/progress.py` | `RunProgress`, `send_status` |
| `logit_tagger/nodes.py` | every node class |
| `web/logit_tagger.js` | the result display below each node |
| `tests-AB/` | measurement harnesses, bootstrapped by `ab_env.py` |

## The tagger pipeline

Ported from ContextAnchoredTileRefine's tile tagger (`context_anchored_tile_refine/tags.py`, its
`tests-AB/tags-bench-log.md` measured precision 0.778 to 0.972).

```
prompt ──> [prompt stage] ──────────────┐   one text-only generate per distinct prompt,
                                        │   kept when grounded in the prompt's words
image ──> [resize] ──> [propose] ──> [merge] ──> [thing check] ──> [verify] ──> [clean] ──> tags
          1 MP cap     greedy, stops  model tags   text-only choice   noul per tag   drop_subsets
                       at 25 tags or  first, then  drops scene,       0.9, or 0.9999
                       a repeat       prompt tags  light, mood words  for prompt-only
```

- Propose and verify run inside the library's `shared_vision_encode`, so the vision tower runs once
  per image.
- The propose stage writes the Qwen chat template itself (`engine.chat`). Core's `thinking=False`
  appends an empty `<think>` block that made Qwen3-VL-4B-Instruct reply empty, refuse, or open with
  `assistant`.
- `engine.clip_generate` calls `comfy.model_prefetch.cleanup_prefetch_queues` after each generate.
  Core's Qwen3 CUDA graph decode keeps each layer's graph bound to the freed KV cache and asserts on
  the next generate in the same node (issue #16441). The cleanup keeps graph decode on, which
  ContextAnchoredTileRefine measured at 100 to 16.5 ms per token.
- `engine.stop_when` wraps core's `sample_token` and returns the stop token once a predicate on the
  complete tags so far holds, since core's decode loop takes no stop condition.
- The CLIP loader types `krea2`, default and `mage` gave identical tags. `flux` and `flux2` load the
  encoder without its vision tower, and `build_classifier` rejects them by name.

## Measured question shapes

On this model through the ComfyUI backend. Scripts: `tests-AB/ab_question_shapes.py` (60 COCO val2017
photos, built by `tests-AB/build_coco60.py`) and `tests-AB/ab_release_gates.py` (72 Krea 2 renders
labeled by two blind judges, 80 labeled prompts).

| Shape | Result |
|---|---|
| image yes/no, question wording | AUC 0.996 on photos, 0.99 or more for every presence, scene and style question on renders, accuracy 0.97 at 0.5 |
| image yes/no, statement wording | "This image contains an orange" read the color: use question wording for free questions |
| text yes/no, "Does this prompt ask for X in the image?" | accuracy 0.972, says no to negated mentions ("no cats") and decoy words ("dogwood") |
| fixed tag list, "Is there X in this image?" | precision 0.971, recall 0.748 at 0.9999 on COCO |
| per-thing missing list | precision 0.976, recall 0.927 at 0.5 on renders |
| whole-prompt yes/no | AUC 0.899 on near-miss prompts only: report per thing instead |
| choice over 8 styles | accuracy 0.875, confidence AUC 0.949 on renders |
| score | ranks brightness at Spearman 0.99, but "sharp" also falls with darkness. Not shipped |
| ranking written prompts | at chance against a blind judge. Not shipped |

One question alone costs about 160 ms per image. Each added packed question costs about 6 ms.

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
- The library's first prime directive bars `generate`, so the generate side of the tagger lives here.
  ContextAnchoredTileRefine holds a second copy of it, which is known duplication.

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
- `tests-AB/ab_env.py` does not yet enable core's dynamic VRAM, so harness generates run without
  graph decode and time about 84 ms per token. ContextAnchoredTileRefine's `ab_env.py` enables it.

## Conventions

- Follow ContextAnchoredTileRefine's patterns: a comfy-free module scope with lazy imports, tooltips
  on every input, errors that name the node and the fix.
- `README.md` is for users and follows the succinct-writing skill.
- A number in a document came from a script in `tests-AB/`, named beside it.
- GPL-3.0, matching the library and the other packs.
- One GPU job at a time on this machine, a 24 GB 3090 Ti shared with ComfyUI.
