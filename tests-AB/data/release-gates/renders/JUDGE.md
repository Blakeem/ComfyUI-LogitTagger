# Judge instructions

You label generated images. Look at each image yourself (open the PNG with your Read tool). Judge only
what is visibly in the image. Do not guess from the prompt.

For each image write one JSON object. Answer every checklist question with "yes", "no" or "unsure".
Use "unsure" whenever a reasonable person could disagree (tiny, cropped, ambiguous or stylized beyond
recognition). A thing counts as present when a viewer would point at it and name it without hesitation.

## Checklist (question id: question)

person: Is there a person (human or clearly humanoid character) in the image?
face: Is a human face clearly visible (eyes, nose and mouth recognizable)?
hand: Is at least one human hand clearly visible?
group: Are there two or more people?
dog: Is there a dog?
cat: Is there a cat?
horse: Is there a horse?
bird: Is there a bird?
vehicle: Is there a car, truck, bus, bicycle, boat, train or other vehicle?
sword: Is there a sword?
food: Is there food?
plant: Is there a tree or other plant?
building: Is there a building or part of one (wall, house, tower)?
water: Is there a body of water (sea, lake, river, pool)?
sky: Is the sky visible?
text: Is there readable text or lettering (letters you could read)?
night: Is it a night scene?
indoors: Is the scene indoors?
photo: Is this a photograph (or photorealistic enough to pass as one)?
anime: Is this in anime or manga style?
painting: Does this look like a painting (oil, acrylic or watercolor)?
render3d: Does this look like a 3D render (CGI)?
pixel: Is this pixel art?
sketch: Is this a pencil or ink sketch or line drawing?
comic: Is this in western comic book style?
monochrome: Is the image black and white or a single color?
smiling: Is there a person who is visibly smiling?

## Intended things

Each image also has a list of intended things from its prompt. For each intended thing answer
"present", "absent" or "unsure" with the same rule.

## Output

Write a JSON file mapping image id to {"checklist": {question id: answer}, "intended": {thing: answer}}.
