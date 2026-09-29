# Judge instructions

You label tags against generated images. Open each image yourself with your Read tool. Judge only what
is visibly in the image. Do not guess from the tag list, since it mixes things the image holds with things
it does not.

For each tag answer one of:

- `present`: a viewer would point at the thing and name it with this tag without hesitation. A part of
  it counts, and a cropped one counts.
- `absent`: the thing is not in the image, or the tag names it wrongly.
- `unsure`: a reasonable person could disagree, since the thing is tiny, hidden, ambiguous or stylized
  beyond recognition.

Judge each tag's words literally.

- A plural tag ("dogs") needs two or more. A singular tag ("dog") is present when one or more are there.
- A count ("three eggs") needs exactly that many visible.
- A collective ("flock of sheep", "bunch of grapes") needs the group, and a pair ("pair of glasses")
  needs the one object.
- A compound ("salt and pepper shakers", "bow and arrow") is present only when every named part is.
- A color word ("orange", "rose", "coral") names the thing, never a color. "orange" needs the fruit.
- A tag with a sign's text ("sign reading OPEN") needs a sign with that text readable.
- A substance ("fog", "smoke", "steam", "sand") needs that substance visible, not only its effect.

## Output

Write one JSON file mapping image id to {tag: answer}, with every tag of every image you were given.
