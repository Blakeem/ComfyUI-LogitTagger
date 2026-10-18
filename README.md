# ComfyUI Logit Tagger

ComfyUI nodes that tag, check and classify images and prompts with the Qwen3-VL text encoder a workflow already loads, such as the one Krea 2 uses. No second model is loaded. Each answer is a probability read from the encoder's logits, so a yes or no answer or a choice needs no generated text. The one exception is the style line that Logit Style Extractor writes for an image.

## Node Selection

| Node | Use when |
|---|---|
| [Logit Tagger](#logit-tagger) | You want tags for each image, such as for a training dataset. |
| [Logit Tag Check](#logit-tag-check) | You want to keep the tags from your own list that each image shows. |
| [Logit Custom Tagger](#logit-custom-tagger) | You want tags that follow your own instruction. |
| [Logit Style Extractor](#logit-style-extractor) | You want the medium and style of an image or a prompt, such as for the start of every tile's prompt. |
| [Logit Image Check](#logit-image-check) | You want a yes or no answer about each image, or to send images down two branches. |
| [Logit Text Check](#logit-text-check) | You want a yes or no answer about each prompt or other text. |
| [Logit Prompt Check](#logit-prompt-check) | You want to know which things a prompt names are missing from each image. |
| [Logit Image Classify](#logit-image-classify) | You want each image sorted into one of your options, such as an art style. |
| [Logit Text Classify](#logit-text-classify) | You want each prompt sorted into one of your options. |

## Installation

Install it through ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes`. The nodes need the `logit-classifier` package, version 0.6.0 or newer, in the Python environment ComfyUI runs in.

```
pip install -U logit-classifier
```

The nodes read the Qwen3-VL text encoder that Krea 2 loads, such as `qwen3vl_4b_fp8_scaled.safetensors`. Load it with Load CLIP and the `krea2` type. The `flux` and `flux2` types drop its vision tower, so the nodes cannot use them.

## The Nodes

### Logit Tagger

The encoder writes candidate tags for each image. Then each tag is scored with a true or false statement about the image, and only the tags the image shows are kept, highest first.

| Input | What it does |
|---|---|
| `prompt` | Optional. The things the prompt names join the candidate tags, and each is kept only where the image shows it. |
| `strict` | On keeps fewer tags and fewer wrong tags. Off keeps more tags. |
| `positions` | On drops a tag that no part of the image shows. It also adds a place to a tag that only one of the three rows or three columns holds. Off keeps the tags as they are. |

An example output with `positions` on is `red car bottom-left, cloud top`.

`tags` holds one comma separated string per image. WD14 Tagger's output has the same shape, so `tags` connects wherever that output connects. Wire `images` and `tags` into Save Image-Text (to Folder) to save each image beside its tags.

### Logit Tag Check

The node checks each tag you give it against each image and keeps only the tags the image shows, highest first. It writes no tags of its own.

| Input | What it does |
|---|---|
| `tags` | The tags to check, separated by commas or new lines. |

The `strict` and `positions` inputs and the outputs work as on [Logit Tagger](#logit-tagger).

### Logit Custom Tagger

The encoder lists the tags your instruction asks for in each image. Then each tag is checked against the image as on Logit Tagger.

| Input | What it does |
|---|---|
| `instruction` | The instruction the encoder follows to list the tags of each image. It starts as Logit Tagger's own instruction, and a blank instruction uses that one. |
| `things_only` | On drops a tag that names no thing to point at, such as a mood, a style or a color, as Logit Tagger does. Turn it off for an instruction that asks for such tags. |

`strict`, `positions` and the outputs are the same as on [Logit Tagger](#logit-tagger).

### Logit Style Extractor

The node writes the medium and the style within it for each image or prompt. When a prompt names a style, its style words are the style line. Otherwise the line is written from the entire image. That line is generated text that no check reads.

| Input | What it does |
|---|---|
| `image` | Optional. Connect the entire image, not its tiles, so that every tile shares one style. |
| `prompt` | Optional. The encoder lists the medium and style words the prompt names, such as oil painting or impressionist. Then a logit classifier keeps the words that name a style. |

Connect an image, a prompt or both. `style` holds one style line per image, or one per prompt when no image is connected. An image or prompt with no style found gets an empty string.

### Logit Image Check

The node asks one yes or no question about each image. A probability of 0.5 or more is a yes.

| Input | What it does |
|---|---|
| `question` | The yes or no question asked about each image. The default is "Is there a person in this image?", and a blank question asks it. Since a statement such as "This image contains an orange" was answered about the color, word your own as a question. |

`yes_images` and `no_images` hold the images of each answer. An empty side skips the nodes wired to it. `answer` is true only when every image answers yes, and `probability` holds one value per image.

### Logit Text Check

The node asks one yes or no question about each text, with no image.

| Input | What it does |
|---|---|
| `text` | The texts to check, such as prompts. |
| `question` | The yes or no question asked about each text. The default is "Does this prompt ask for a person or people in the image?", and a blank question asks it. |

The default's wording answers no to a prompt that says "no cats", where "Does this text mention a cat?" answers yes. "A person" alone answered no to 9 of 12 prompts of a group, so the default names both (`ab_person_wording.py`).

`yes_texts` and `no_texts` hold the texts of each answer. The other outputs work as they do on Logit Image Check.

### Logit Prompt Check

The encoder lists the physical things the prompt names. Then each image is asked whether it shows each thing, and a thing that scores below 0.01 is missing.

| Input | What it does |
|---|---|
| `prompt` | The prompt the images were made from. |

`missing` holds the missing things of each image, and `score` holds the share of things each image shows. `complete_images` and `incomplete_images` sort the images by whether any thing is missing. `images` returns every image beside its `missing` text.

### Logit Image Classify

The node picks one of the options for each image. The default question and options sort images into 8 art styles.

| Input | What it does |
|---|---|
| `question` | The question the options answer, asked about each image. A blank question asks the default one. |
| `options` | 2 to 51 answers, separated by commas. A blank widget uses the default art styles. |

`choice` is the option as you typed it, and `index` is its position, counting from 0. `confidence` is 0 when every option is equally likely and 1 when one option takes all the probability. `images` returns each image beside its choice.

### Logit Text Classify

The node picks one of the options for each text, such as a prompt. The default question and options sort prompts into the 8 art styles of Logit Image Classify.

| Input | What it does |
|---|---|
| `text` | The texts to classify, such as prompts. |
| `question` | The question the options answer, asked about each text. A blank question asks the default one. |

`texts` returns each text beside its choice. The `options` input and the other outputs work as on [Logit Image Classify](#logit-image-classify).

## Batches and Prompts

- Every image input takes an image batch or a list of batches, and each image is answered on its own. The list outputs keep the image order.
- A `prompt` or `tags` input takes one text for every image, or one text per image.
- A prompt in another language is translated to English before its things are read.
- One progress bar covers the entire batch. The status line below it shows the current image and stage, then how long the run took.
- Each node shows its results below its inputs.
- Wire `answer` or `all_present` into core's If/Else Switch to pick a branch. The switch skips a core node on the other branch unless an output node, such as Save Image or Preview Image, reads that node directly or through other nodes. Since every pack node is also an output node, a pack node on the other branch still runs. It is skipped only when one of its inputs comes from an empty yes or no output, directly or through other nodes. Those outputs are `yes_images` and `no_images`, `yes_texts` and `no_texts`, and `complete_images` and `incomplete_images`.

## Example Workflows

The `workflows` folder holds eleven workflows, and ComfyUI lists them in its template browser.

| Workflow | What it does |
|---|---|
| [Tag an image](workflows/Tag%20an%20image.json) | Tags one image. Its Prompt node connects to the `prompt` input. |
| [Tag a dataset](workflows/Tag%20a%20dataset.json) | Tags every image in a folder and saves each image beside a text file of its tags. |
| [Check a tag list](workflows/Check%20a%20tag%20list.json) | Checks a list of tags against one image and keeps the tags the image shows. |
| [Tag clothing with your own instruction](workflows/Tag%20clothing%20with%20your%20own%20instruction.json) | Tags the clothing and accessories in one image with an instruction that asks for them. |
| [Read the style of an image or a prompt](workflows/Read%20the%20style%20of%20an%20image%20or%20a%20prompt.json) | Writes the style line of one image and lists the style words of one prompt. |
| [Sort images by a question](workflows/Sort%20images%20by%20a%20question.json) | Saves the images that answer yes and the images that answer no to separate folders. |
| [Sort images by art style](workflows/Sort%20images%20by%20art%20style.json) | Saves each image beside the art style it shows. |
| [Check prompt texts](workflows/Check%20prompt%20texts.json) | Sorts four prompts into those that ask for a person and those that do not. |
| [Sort prompts by art style](workflows/Sort%20prompts%20by%20art%20style.json) | Picks an art style for each of four prompts that do not name one. |
| [Render and check the prompt](workflows/Render%20and%20check%20the%20prompt.json) | Renders a prompt with Krea 2 and saves the render only when it shows every thing the prompt names. |
| [Upscale and fix a face with SD 1.5](workflows/Upscale%20and%20fix%20a%20face%20with%20SD%201.5.json) | Renders a figure with SD 1.5 and upscales it 2x, with a style line and tags for each tile. Then it refines the face when Logit Image Check finds one. |

The Upscale and fix a face with SD 1.5 workflow also needs the ComfyUI Inpaint Crop and Stitch pack (`comfyui-inpaint-cropandstitch`), the `v1-5-pruned-emaonly-fp16` checkpoint, the `4x-UltraSharp` upscale model and the `sam3.1_multiplex_fp16` checkpoint for core's SAM3 Detect.

## Accuracy and Speed

The tests use 72 Krea 2 renders and 80 prompts labeled by blind judges, plus COCO photos. The times come from an RTX 3090 Ti with the fp8 encoder, and the scripts are in `tests-AB`.

| Node | Accuracy | Time per image | Script |
|---|---|---|---|
| Logit Tagger | 95% of kept tags judged correct, 9.7 to 14.0 correct tags per image | 1.6 to 3.4 s | `ab_tagger_compare.py` |
| Logit Tagger with `strict` | 96 to 98% of kept tags judged correct, 5.1 to 9.4 correct tags per image | as above | `ab_tagger_score.py` |
| Logit Tag Check | 90% precision and 76% recall over the 80 COCO categories, 95% and 42% with `strict`. Most wrong tags were "orange" on an orange colored thing. | 0.7 s | `ab_release_gates.py` |
| Logit Style Extractor from a prompt | The kept words name the medium of 85% of the 72 render prompts, with 0.053 wrong phrases per prompt over all 152 prompts | not measured | `ab_style.py` |
| Logit Style Extractor from an image | The line names the medium of 49 of the 72 renders, and on 28 of the 72 it names content such as a robot | 1.55 s on a 1 MP render | `ab_style.py` |
| Logit Image Check | 97% over 27 questions | 0.28 s | `ab_release_gates.py` |
| Logit Text Check | 97% over 80 prompts | 0.1 s | `ab_release_gates.py` |
| Logit Prompt Check | 98% precision and 93% recall on missing things | 0.39 s, plus about 1 s per prompt | `ab_release_gates.py`, `ab_tagger_compare.py` |
| Logit Image Classify | 92% over 8 art styles | 0.28 s | `ab_release_gates.py` |
| Logit Text Classify | 72 of the 72 render prompts over 8 art styles, 51 of 72 with each prompt's medium phrase removed | not measured | `ab_text_classify.py` |

Logit Custom Tagger has no row, since its result depends on the instruction.

## License

GPL-3.0
