# ComfyUI Logit Tagger

A ComfyUI node that tags images with the Qwen3-VL text encoder a workflow already loads, such
as the one Krea 2 uses. So no second model is loaded.

The encoder writes candidate tags for each image. Then every tag is checked against the image,
and only the tags the image shows are kept.

## Install

Install it through ComfyUI Manager, or clone this repository into `ComfyUI/custom_nodes`. The
node needs the `logit-classifier` package, version 0.2.0 or newer, in the Python environment
ComfyUI runs in.

```
pip install -U logit-classifier
```

## Node Inputs and Outputs

| Input | Description |
|---|---|
| `clip` | a Qwen3-VL text encoder, such as Load CLIP with the `krea2` type |
| `image` | the images to tag, one at a time |
| `prompt` | optional text, such as the prompt the image was made from |

The `krea2` type and the default type give the same tags. The `flux` and `flux2` types load
the encoder without its vision tower, so the node rejects them.

`tags` is one comma separated string per image, as a list. The node also shows the tags below
its inputs.

## How It Works

1. The image is scaled down to about 1 megapixel.
2. The encoder writes the things it sees as comma separated tags.
3. Each tag is asked as a yes or no question, "This image visibly contains" plus the tag. The
   answer is a probability read from the encoder's logits at one token position, so no text is
   written in this step. Every tag is checked in the same pass.
4. The tags with a probability of 0.5 or more are kept, highest first.

The prompt sets the words the tags use. A prompt object the image lacks is still proposed, and
it fails the check. We tested a prompt that named a white wolf, a lantern and a sword on an
image with none of them. Each scored 0.0002 or less.

## Speed

A 1 megapixel image takes about 5 to 7 seconds on an RTX 3090 Ti, and a dense image that
writes over 40 tags took 16 seconds. Writing the tags takes most of it, and checking them
takes under a second whatever the tag count.
`tests-AB/ab_tagger.py` measured these times.

## License

GPL-3.0
