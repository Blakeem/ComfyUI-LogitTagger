# Tag judge rubric

You grade image tags. Each entry of to_judge.json holds an image path, a key and a list of tags. Open
the image with your Read tool and look at it carefully (zoom by reading it again if needed). Judge each
tag only against what is visible. You do not know which system wrote which tag.

Grades:
- "yes": the tag names something clearly visible in the image, and a viewer would agree the tag fits.
  Color, material and count words in the tag must also be right ("red car" on a blue car is "no").
- "no": the thing is not in the image, or a word in the tag is wrong about it.
- "vague": the tag is true but names no specific thing a viewer could point at, or is too generic to be
  useful as an image tag ("details", "scene", "objects", "texture", "background", "setting", "lighting",
  "colors", "atmosphere").
- "unsure": a reasonable person could disagree (tiny, cropped, ambiguous or stylized beyond recognition).

Write JSON {key: {tag: grade}} covering every tag of every image you were assigned.
