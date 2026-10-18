# Style judge rubric

You grade text that a system wrote about how an image is made. Read `to_judge.json` in your assigned run folder.
You do not know which system wrote which phrase or line. Grade only from the text you are given.

## Prompt phrases

An entry with `phrases` holds a key, the image prompt and phrases copied from that prompt. Grade each phrase by what
it names in that prompt.

- "yes": the phrase names how the image is made. That is its medium (photograph, oil painting, watercolor, pencil
  sketch, 3D render, pixel art, comic book, anime illustration, a print, charcoal), a drawing or painting technique
  (impasto, cross-hatching, halftone shading, bold ink lines, cel shading), an art movement or period style, an
  artist's style, a rendering or game look ("16-bit retro game style"), or a film or camera look (film grain, bokeh,
  macro photography). A phrase that names a medium beside a shot type or a format ("close-up photograph", "comic
  book panel") is "yes".
- "no": the phrase names what the image shows or how its scene looks. That is an object or the material of an
  object, a person, an animal, a place, an action, a time of day, weather, a color or a palette, lighting, a mood,
  a genre or theme of the scene (fantasy, steampunk), a shot type or composition alone ("close-up", "wide shot"), or
  a quality word ("masterpiece", "best quality", "highly detailed"). A medium word that names a thing in the scene,
  such as "a painting on the wall" or "a drawing class", is "no".
- "unsure": a reasonable person could disagree.

## Style lines

An entry with `line` holds a key, one line written about an entire image, and that image's medium. Give two
grades.

The "line" grade:
- "yes": the line names only how the image is made, its medium and the style within that medium, with words about
  technique, color handling, rendering or lighting style.
- "no": the line also names something the image shows, such as an object, a person, an animal, a place or an
  action ("a robot", "a Venetian canal", "a woman reading").
- "unsure": a reasonable person could disagree.

The "medium" grade:
- "yes": the line names the entry's medium in any wording ("photographic" for photograph, "graphite drawing" for
  pencil sketch).
- "no": the line names another medium, or none.
- "unsure": a reasonable person could disagree.

## Output

Write JSON `{key: {phrase: grade}}` for prompt phrases and `{key: {"line": grade, "medium": grade}}` for style lines, covering every
phrase or line of every entry, to the file name you were given.
