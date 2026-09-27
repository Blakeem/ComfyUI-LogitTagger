# ComfyUI Logit Tagger, project guide

A ComfyUI node pack built on the `logit-classifier` library. This file covers the nodes that
exist or are being built. Ideas for later nodes live in `TODO.md` and nowhere else.

## Nodes

| Node | Status | Inputs | Output |
|---|---|---|---|
| Logit Tagger | built, not released | `clip`, `image` batch, `prompt` | one tag string per image, as a list |

## The Logit Tagger pipeline

Five stages run per image, in order. Only the tags that leave the filter stage reach the
output and the display.

```
image batch ──> [resize stage] ──> [propose stage] ──> [parse stage] ──> [verify stage] ──> [filter stage] ──> STRING list + node display
                 one image,         clip.generate,      CSV text to        one noul per       keep tags at or
                 1 MP cap           greedy, no think,   unique candidate   candidate, one     above the noul
                                    stops on a repeat,  tags               request per image  threshold
                                    prompt anchors
                                    the vocabulary
```

| Stage | Where |
|---|---|
| every stage, `tag_image` (one image, returns `TagDetail`), `tag_images` (the batch) | `logit_tagger/tagging.py` |
| the node contract, progress bar, interrupt check | `logit_tagger/node.py` |
| the tag display below the inputs | `web/logit_tagger.js` |
| the harness over the real model | `tests-AB/ab_tagger.py`, bootstrapped by `tests-AB/ab_env.py` |

- The propose stage writes the Qwen chat template itself (`PROPOSE_TEMPLATE`) and never adds a
  `<think>` block. Core's `thinking=False` appends an empty one, and on Qwen3-VL-4B-Instruct
  that made the reply open with `assistant`, come back empty, or refuse.
- The propose stage's generate runs inside `cuda_graphs_disabled()`. Core's Qwen3 CUDA graph
  decode (PR #15623, 2026-09-18) raises a device side assert on the second multi step image
  generate in one process. Core calls alone reproduce it, and core's own Generate Text node
  meets it too.
- The propose stage's generate also runs inside `stop_on_repeat()`. It wraps core's
  `sample_token` and returns the model's stop token once the reply writes a block of up to 4 tags
  twice back to back (`MAX_REPEAT_BLOCK`), since core's decode loop takes no stop condition.
  Greedy decode loops on some images whatever the wording, in `tests-AB/ab_propose_prompts.py`.
- The verify stage costs 0.27 to 0.79 s per image whatever the tag count, since the library
  packs every tag behind one image prefix into one forward pass. Before that it cost about
  540 ms per tag. `tests-AB/ab_packed_parity.py` compares the packed pass with the per tag
  path it falls back to. The propose stage, 4.6 to 15 s, is now the entire remaining cost.
- The CLIP loader types `krea2`, default and `mage` gave identical tags. `flux` and `flux2`
  load the encoder without its vision tower, and `build_classifier` rejects them by name.

- The propose stage generates text. The verify stage generates nothing. It reads one logit row
  per candidate through the library's `Classifier`, over the CLIP the workflow already holds.
- The prompt input reaches the propose stage only. It names the vocabulary the candidates use.
  The verify stage judges every candidate against the image alone, so a prompt object the image
  lacks fails there.
- Every tuning value (instruction text, token budget, pixel cap, candidate cap, threshold) is a
  backend constant set by a measurement in `tests-AB/`. None is a widget.

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
`E:\logit-classifier\src` is live on the next ComfyUI start with no sync step or copy.

```
.venv\Scripts\uv.exe pip install -e E:\logit-classifier --python .venv\Scripts\python.exe
```

A Registry install gets the published wheel instead, through `[project] dependencies`.

## Dependency rules

- `[project] dependencies` names the bare `logit-classifier`, never an extra. The `[hf]` extra
  names torch, and pip resolving torch beside ComfyUI's pinned CUDA build breaks the install.
- Every import of `logit_classifier` is lazy, inside the function that needs it. A missing
  library raises an error that prints the pip command. That covers a user who installed the
  pack by hand.
- The library's ComfyUI backend holds no `comfy` import. It duck types the CLIP.
- A bug in the library is fixed in `E:\logit-classifier` and not patched around here. The owner
  publishes the fix to PyPI.

## Verified against the installed core

Read from the server source at commit `c194dd00` on 2026-09-22.

| Fact | Where |
|---|---|
| Krea 2 loads the entire Qwen3-VL-4B, lm_head mapping included | `comfy/sd.py:1943` |
| the 4B ties its word embeddings, so logits come from `embed_tokens` | `comfy/text_encoders/llama.py:1237`, `BaseQwen3.logits` |
| `BaseGenerate.generate` computes logits and hands them to `self.sample_token`, the capture point | `llama.py:1164-1170` |
| greedy sampling is `argmax` when `do_sample` is False | `llama.py:1228` |
| a text starting with `<\|im_start\|>`, or `skip_template=True`, bypasses the chat template | `comfy/text_encoders/qwen3vl.py:167` |
| `thinking=False` appends the empty `<think>` block that suppresses reasoning | `qwen3vl.py:184` |
| Krea 2's tokenizer defaults `thinking=True`, so the propose stage passes `thinking=False` explicitly | `comfy/text_encoders/krea2.py:28` |
| the tokenizer replaces each `<\|image_pad\|>` (151655) with the next image in `images` | `qwen3vl.py:190-195` |
| the image path builds `visual_pos_masks` at batch 1, so one image per generate call | `qwen3vl.py:80` |
| core resizes a VL image only above 12.8 MP, so a 4k input costs about 12k tokens unless the node resizes first | `comfy/text_encoders/qwen_vl.py:9-46` |
| the CLIP compatibility gate is `hasattr(clip, "generate")` and `hasattr(clip, "decode")` | CATR `captions.py:690` |
| frontend 1.53.6 exposes `window.comfyAPI.textPreviewWidgets` (`addTextPreviewWidgets`, `updateTextPreviewWidgets`), which core's Preview as Text uses | `comfyui_frontend_package/static/assets/core-*.js` |
| Preview as Text returns `{"ui": {"text": (value,)}, "result": (value,)}` | `comfy_extras/nodes_preview_any.py` |

## Tests and gates

- Unit gate: `.venv\Scripts\python.exe -m pytest tests -q` from the pack root, 76 tests, no GPU.
- Lint gate: `uvx ruff@0.16.2 check .` reports zero findings.
- Harness: `.venv\Scripts\python.exe tests-ABb_tagger.py <image>... [--prompt] [--threshold]
  [--instruction]` prints the proposal, every candidate's probability, the kept tags and the
  stage times. Check `nvidia-smi` is idle first.

## Conventions

- Follow ContextAnchoredTileRefine's patterns: the V1 node contract (`INPUT_TYPES`,
  `RETURN_TYPES`, `FUNCTION`), a comfy-free module scope with lazy imports, tooltips on every
  input, errors that name the node and the fix.
- `README.md` is for users and follows the succinct-writing skill.
- A number in a document came from a script in `tests-AB/`, named beside it.
- GPL-3.0, matching the library and the other packs.
- One GPU job at a time on this machine, a 24 GB 3090 Ti shared with ComfyUI.
