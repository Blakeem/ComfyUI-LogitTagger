# To do

## Measured follow ups

- Several questions per check node. Logit Image Check could take one question per line and ask them in one packed
  request per image, with a rule that sends an image to yes when every question or any question answers yes. Four
  questions would cost about one question's time instead of four. The render-check probe already asks 27 questions
  packed per image. Open: a combined decision line in `tests-AB/ab_release_gates.py`, and a measurement of the
  attribute questions a rejection list needs, such as watermark, blur and cropped subject.
- Cross-run caches. The prompt stage could be cached per prompt and the propose reply per image, bound to the CLIP
  object the way ContextAnchoredTileRefine's `ClipBoundCache` works. A queued seed on one prompt would skip about
  1 s in Logit Prompt Check, and a re-run of the Tagger on unchanged images would skip each propose. A decision matrix
  scored it within the margin of error against the module state it adds, so it waits for a user who needs it.
- The Tagger's pixel cap. The check nodes read images at 0.5 MP. The Tagger still reads them at 1 MP, since its
  propose and verify share one vision encode. An arm in `tests-AB/ab_tagger_compare.py` at 0.5 MP would show whether
  the tags hold.
- Anchored proposals. With a prompt connected, the propose stage copied the prompt's objects and added few of its
  own in `tests-AB/ab_tagger.py`. A wording that asks for the prompt's words plus anything else visible is worth
  testing.
- Template drift. Wordings other than the current one drifted into one word patterns, such as `market goods, market
  items, market setup`. A 6 tag cycle of these passes the repeat stop.
- Core issue #16441. Core's Qwen3 CUDA graph decode asserts on the next generate in one node. Once core fixes it, the
  `cleanup_prefetch_queues` call in `engine.clip_generate` can go.

## Node ideas

Each idea names the job and what is still open. The built nodes cover Text Gate (Logit Text Check), Image Gate and
Batch Filter (Logit Image Check), Declared Tags (the Tagger's `tags` input) and Choice Router (Logit Image Classify).

- Tag Text. Text in, a verified tag list out, the Tagger without the image. Use: bucketing danbooru tag text locally.
  Open: no measurement of text-only tag verification yet.
- Ask (JSON). A System One request body in, the response JSON out, for a workflow that builds its questions
  elsewhere.
- Ground Prompt. A prompt, keywords and an image in, the prompt fragments the image holds out. Tracked as stage 5 in
  the library's `TODO.md`.
- Classifier Loader. Loads a Qwen through the library's `[hf]` extra for a workflow with no vision language encoder.
  Open: memory management for a second resident model.

## Measured and dropped

- Image Score and Best of N. A score question ranked brightness at Spearman 0.99, but "sharp" also fell with darkness
  (`tests-AB/ab_question_shapes.py`).
- Prompt Brainstorm and Rank. Ranking written prompts was at chance against a blind judge
  (`tests-AB/ab_prompt_rank.py`).
- Repetition penalty on propose. Core's penalty at 1.05 changed the kept tags on 13 of 17 images and still looped on
  one market image (`tests-AB/ab_propose_loops.py`).
