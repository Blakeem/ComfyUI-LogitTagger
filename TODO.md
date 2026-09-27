# To do

## Logit Tagger follow ups

- Propose speed. Writing the tags now takes nearly all of an image's time, 4.6 to 15 s,
  since core's decode runs about 87 ms per token with CUDA graphs off. Once core fixes issue
  #16441, the CUDA graph guard can go and decode may speed up.
- Category echo. The instruction's category list (objects, people, animals, clothing,
  materials, setting) reached the kept tags 9 times on 28 images in
  `tests-AB/ab_propose_prompts.py`, mostly on market images. Every wording without the list
  wrote longer lists and ran slower, so a wording that keeps the list's scope without nouns the
  reply can copy is worth testing.
- Template drift. Wordings other than the current one drifted into one word patterns, such as
  `market goods, market items, market setup` or `fantasy city, fantasy street`. A 6 tag cycle
  of these passes `MAX_REPEAT_BLOCK`.
- Repetition penalty. In `tests-AB/ab_propose_loops.py` core's penalty at 1.05 changed the kept
  tags on 13 of 17 images, lost 22 of 35 tags on one city, and still looped on one market
  image, so it is not used.
- Anchored proposals. With a prompt connected, the propose stage copied the prompt's objects
  and added few of its own in `tests-AB/ab_tagger.py`. A wording that asks for the prompt's
  words plus anything else visible is worth testing.
- Core bug. Core's Qwen3 CUDA graph decode (PR #15623) asserts on the second image generate
  in one process. It is reported as Comfy-Org/ComfyUI issue #16441, and PR #16476 is an open,
  unreviewed fix. The node turns CUDA graphs off around its own generate until a fix lands.

## Node ideas

Ideas for nodes after Logit Tagger. None is designed or committed. Each entry names the job,
the library question type it rests on, and what is still open.

The library answers three question types. A choice picks one of several options. A score places
the input on an ordered scale and returns a float. A noul returns the probability that one
statement is true.

### From the owner

#### Text Gate

Text in, a declared question, and the text out unchanged beside a boolean and a probability.
Rests on one noul. Uses: skip a render when the prompt is a greeting, moderate a prompt on a
shared instance, hold a render until an article meets a condition.

Open: whether the node also offers core's `ExecutionBlocker` stop beside the boolean.

#### Image Gate

The same as Text Gate over an image. Uses: route an image with a face to a face fix, stop a
save when the image fails a safety question, send a photo and an illustration to different
branches.

#### Prompt Brainstorm and Rank

A short request in, a batch of expanded prompts out, ranked by a declared criterion such as
"the most epic". Two parts. The brainstorm part generates variants with sampling on. The rank
part asks one score question per variant and sorts by the float. A second form ranks the
rendered images rather than the prompts, which is Best of N below.

Open: whether ranking text predicts a better image at all. Measure it before building the
brainstorm part.

#### Ask (JSON)

A System One request body in, the response JSON out. The open node that exposes every question
type the library supports, for a workflow that builds its questions elsewhere.

### Added in review

#### Tag Text

Text in, a verified tag list out, the Logit Tagger pipeline without the image. Use: bucketing
danbooru tag text locally. The one text classifier with real installs posts tags to a paid API
and parses its answer out of a markdown fence.

#### Declared Tags

An image and a user list of tags in, the tags that pass a threshold and a probability per tag
out. Logit Tagger proposes its own candidates, and this node checks a fixed list. Uses: dataset
filtering against a rejection list such as watermark, blur, or cropped subject, and filing into
folders.

#### Choice Router

An image or text and a list of options in, the winning option as a string and its index as an
int out. Rests on one choice question. Feeds core's switch or a combo for routing to more than
two branches.

#### Image Score

An image and an ordered list of level descriptions in, a float out. Rests on one score
question. Use: a judged dial, such as how much detail is missing driving the denoise value.

#### Best of N

An image batch and one criterion in, the best image and its index out. Rests on one score
question per image. Use: render several seeds and keep the winner, or a judge loop that
re-rolls below a line.

#### Batch Filter

An image batch and one noul question in, the images that pass out. Use: cleaning a training set
inside a workflow.

#### Ground Prompt

A prompt, keywords, and an image in, the prompt fragments the image holds out. The grounded
prompt surface for ContextAnchoredTileRefine. Tracked as stage 5 in the library's `TODO.md`.

#### Classifier Loader (weights)

Loads a Qwen through the library's `[hf]` extra for a workflow with no vision language encoder.
The text only `Qwen3-4B-Instruct-2507` scores higher on noul. Open: memory management for a
second resident model.

### Notes for the gate and router nodes

- One state and many questions in one call. A question costs 141 ms alone and 8 questions cost
  142 ms together, measured by the library over the transformers backend, because the state is
  prefilled once. A node taking one question per call throws that away.
- Core's If/Else Switch skips the branch it does not take, since only the input
  `check_lazy_status` returns is made a strong link (`execution.py:504-522`). A switch from
  another pack runs both branches, so a gate wired to it saves no compute. Ship an example wired
  to core's switch and say why in the README.
- Core's `ExecutionBlocker` blocks every consumer, silently when its message is None, and core
  says to prefer a lazy input (`comfy_execution/graph_utils.py:140`).
- Issue #16407 of 2026-09-18 shows a required input upstream of core's switch still forcing
  validation. PR #16417 of 2026-09-20 is open to add a pass through. Test against core and
  against `StableLlama/ComfyUI-basic_data_handling`.
- Measure the noul threshold per question shape. Noul saturates near 0 and 1 under the
  temperature fitted on choice questions. The library measured its best threshold on text near
  0.9999, while Logit Tagger's image tags split at 0.23 and below or 0.80 and up, so 0.5 works
  there (`tests-AB/ab_tagger.py`).
