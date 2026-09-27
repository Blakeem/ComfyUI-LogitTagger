# ComfyUI Logit Tagger

ComfyUI nodes that tag, check and classify images with the Qwen3-VL text encoder a workflow already loads, such as the one Krea 2 uses. No second model is loaded. Each answer is a probability read from the encoder's logits, so a yes or no answer or a choice needs no generated text.

## Node Selection

| Node | Use when |
|---|---|
| [Logit Tagger](#logit-tagger) | You want tags for each image, such as for a training dataset. |
| [Logit Image Check](#logit-image-check) | You want a yes or no answer about each image, or to send images down two branches. |
| [Logit Text Check](#logit-text-check) | You want a yes or no answer about each prompt or other text. |
| [Logit Prompt Check](#logit-prompt-check) | You want to know which things a prompt names are missing from each image. |
| [Logit Image Classify](#logit-image-classify) | You want each image sorted into one of your options, such as an art style. |

## Installation

Install it through ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes`. The nodes need the `logit-classifier` package, version 0.3.0 or newer, in the Python environment ComfyUI runs in.

```
pip install -U logit-classifier
```

The nodes read the Qwen3-VL text encoder that Krea 2 loads, such as `qwen3vl_4b_fp8_scaled.safetensors`. Load it with Load CLIP and the `krea2` type. The `flux` and `flux2` types drop its vision tower, so the nodes cannot use them.

## The Nodes

### Logit Tagger

The encoder writes candidate tags for each image. Then each tag is asked as a yes or no question about the image, and only the tags the image shows are kept, highest first.

| Input | What it does |
|---|---|
| `prompt` | Optional. The things the prompt names join the candidate tags, and each is kept only where the image shows it. |
| `instruction` | Optional. This text replaces the built-in request that asks the encoder to list what each image holds. |
| `tags` | Optional. The node checks only these tags against each image and writes none of its own. |

`tags` holds one comma separated string per image. Wire `images` and `tags` into Save Image-Text (to Folder) to save each image beside its tags.

### Logit Image Check

The node asks one yes or no question about each image. A probability of 0.5 or more is a yes. The default question is "Is there a person in this image?". Since a statement such as "This image contains an orange" was answered about the color, word your own as a question.

`yes_images` and `no_images` hold the images of each answer. An empty side skips the nodes wired to it. `answer` is true only when every image answers yes, and `probability` holds one value per image.

### Logit Text Check

The node asks one yes or no question about each text, with no image. The default question is "Does this prompt ask for a person in the image?". That wording answers no to a prompt that says "no cats", where "Does this text mention a cat?" answers yes.

`yes_texts` and `no_texts` hold the texts of each answer. The other outputs work as they do on Logit Image Check.

### Logit Prompt Check

The encoder lists the physical things the prompt names. Then each image is asked whether it shows each thing, and a thing below 0.5 is missing.

`missing` holds the missing things of each image, and `score` holds the share of things each image shows. `complete_images` and `incomplete_images` sort the images by whether any thing is missing. `images` returns every image beside its `missing` text.

### Logit Image Classify

The node picks one of the options for each image. The default question and options sort images into 8 art styles. Separate 2 to 51 options with commas.

`choice` is the option as you typed it, and `index` is its position, counting from 0. `confidence` is 0 when every option is equally likely and 1 when one option takes all the probability. `images` returns each image beside its choice.

## Batches and Prompts

- Every node takes an image batch or a list of batches and answers each image on its own. The list outputs keep the image order.
- A prompt or text input takes one text for every image, or one text per image.
- A prompt in another language is translated to English before its things are read.
- One progress bar covers the entire batch. The status line below it shows the current image and stage, then how long the run took.
- Each node shows its results below its inputs.
- Core's If/Else Switch runs only the branch it takes. So wire `answer` or `all_present` to it to skip the work on the other branch.

## Example Workflows

The `example_workflows` folder holds four workflows, and ComfyUI lists them in its template browser.

| Workflow | What it does |
|---|---|
| Tag a dataset | Tags every image in a folder and saves each image beside a text file of its tags. |
| Sort images by a question | Saves the images that answer yes and the images that answer no to separate folders. |
| Render and check the prompt | Renders a prompt with Krea 2 and saves the render only when it shows every thing the prompt names. |
| Sort images by art style | Saves each image beside the art style it shows. |

## Accuracy and Speed

The tests use 72 Krea 2 renders and 80 prompts labeled by blind judges, plus COCO photos. The times come from an RTX 3090 Ti with the fp8 encoder, and the scripts are in `tests-AB`.

| Node | Accuracy | Time per image | Script |
|---|---|---|---|
| Logit Tagger | 95% of kept tags judged correct, 9.7 to 14.0 correct tags per image | 1.6 to 3.4 s | `ab_tagger_compare.py` |
| Logit Tagger with `tags` | 97% precision and 75% recall over the 80 COCO categories | 0.5 s | `ab_release_gates.py` |
| Logit Image Check | 97% over 27 questions | 0.28 s | `ab_release_gates.py` |
| Logit Text Check | 97% over 80 prompts | 0.1 s | `ab_release_gates.py` |
| Logit Prompt Check | 97% precision and 95% recall on missing things | 0.39 s, plus about 1 s per prompt | `ab_release_gates.py`, `ab_tagger_compare.py` |
| Logit Image Classify | 89% over 8 art styles | 0.29 s | `ab_release_gates.py` |

## License

GPL-3.0
