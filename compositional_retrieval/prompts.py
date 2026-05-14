"""Human-readable phrases and prompt templates used by Method 3 (Prompt Ensembling v2).

`humanized_mappings_pos` and `humanized_mappings_neg` are person-referring
phrases for the positive and negative side of each CelebA attribute. We deliberately
keep linguistic negatives (e.g. "clean-shaven" vs "bearded") rather than relying
on "not X" — CLIP's text encoder attends to the object token regardless of "not",
so phrasing actually matters.

`prompt_templates_v2` is CLIP's official ImageNet 80-template set (the canonical
zero-shot ensemble) plus a small number of portrait-specific templates that fit
CelebA's domain.
"""

from __future__ import annotations


humanized_mappings_pos: dict[str, list[str]] = {
    "5_o_Clock_Shadow":     ["a person with a 5 o'clock shadow", "a person with light facial stubble", "a person with short beard stubble", "a face with a 5 o'clock shadow", "a man with a 5 o'clock shadow", "a person with visible beard stubble"],
    "Arched_Eyebrows":      ["a person with arched eyebrows", "a person with curved eyebrows", "a face with high arched eyebrows", "a portrait with strongly arched eyebrows", "a person whose eyebrows are clearly arched"],
    "Attractive":           ["an attractive person", "a good-looking person", "a visually appealing person", "a beautiful person", "an attractive face", "a strikingly attractive person"],
    "Bags_Under_Eyes":      ["a person with bags under the eyes", "a person with eye bags", "a tired-looking person with under-eye bags", "a face with visible under-eye bags", "a portrait of a person with bags under the eyes"],
    "Bald":                 ["a bald person", "a person with no hair", "a person with a shaved head", "a person with a completely bald head", "a man who is bald", "a portrait of a bald person"],
    "Bangs":                ["a person with bangs", "a person with fringe hair", "a face with bangs across the forehead", "a portrait of someone with bangs", "a person whose hair has bangs"],
    "Big_Lips":             ["a person with full lips", "a person with big lips", "a face with prominent lips", "a portrait of a person with very full lips"],
    "Big_Nose":             ["a person with a big nose", "a person with a large nose", "a face with a prominent nose", "a portrait of a person with a noticeably big nose"],
    "Black_Hair":           ["a person with black hair", "a person with dark black hair", "a portrait of a black-haired person", "a person whose hair is black"],
    "Blond_Hair":           ["a person with blond hair", "a person with blonde hair", "a person with light blonde hair", "a portrait of a blond person", "a person whose hair is blonde"],
    "Blurry":               ["a blurry photo of a person", "an out-of-focus image of a person", "a blurred image of a person", "a low-quality blurry portrait", "a defocused photograph of a face"],
    "Brown_Hair":           ["a person with brown hair", "a person with dark brown hair", "a portrait of a brown-haired person", "a person whose hair is brown"],
    "Bushy_Eyebrows":       ["a person with bushy eyebrows", "a person with thick eyebrows", "a face with very thick eyebrows", "a portrait of a person with bushy eyebrows"],
    "Chubby":               ["a chubby person", "a person with a round face", "a person with a chubby face", "a portrait of a chubby person"],
    "Double_Chin":          ["a person with a double chin", "a person with a noticeable double chin", "a face with a clear double chin", "a portrait of a person with a double chin"],
    "Eyeglasses":           ["a person wearing eyeglasses", "a person wearing glasses", "a person with glasses", "a face with eyeglasses", "a portrait of a person wearing glasses", "a person who wears glasses"],
    "Goatee":               ["a person with a goatee", "a person with a goatee beard", "a man with a goatee", "a portrait of a person with a goatee"],
    "Gray_Hair":            ["a person with gray hair", "a person with grey hair", "a person with silver hair", "a portrait of a gray-haired person", "an older person with gray hair"],
    "Heavy_Makeup":         ["a person wearing heavy makeup", "a person with noticeable makeup", "a person with strong makeup", "a face with heavy makeup", "a portrait of a person wearing heavy makeup"],
    "High_Cheekbones":      ["a person with high cheekbones", "a person with prominent cheekbones", "a face with sharply defined high cheekbones"],
    "Male":                 ["a man", "a male person", "a portrait of a man", "a photograph of a man", "a male face"],
    "Mouth_Slightly_Open":  ["a person with their mouth slightly open", "a person with slightly open lips", "a face with parted lips", "a portrait of a person whose mouth is slightly open"],
    "Mustache":             ["a person with a mustache", "a person with facial hair and a mustache", "a man with a mustache", "a portrait of a person with a mustache"],
    "Narrow_Eyes":          ["a person with narrow eyes", "a person with small eyes", "a face with narrow eyes", "a portrait of a person with narrow eyes"],
    "No_Beard":             ["a clean-shaven person", "a person without a beard", "a person with no facial hair", "a portrait of a clean-shaven person", "a face without any beard"],
    "Oval_Face":            ["a person with an oval face", "a person with an oval-shaped face", "a portrait of a person with an oval face"],
    "Pale_Skin":            ["a person with pale skin", "a person with light skin tone", "a portrait of a person with pale skin", "a face with very pale skin"],
    "Pointy_Nose":          ["a person with a pointy nose", "a person with a sharp nose", "a face with a pointy nose"],
    "Receding_Hairline":    ["a person with a receding hairline", "a person with thinning hairline", "a portrait of a person whose hairline is receding"],
    "Rosy_Cheeks":          ["a person with rosy cheeks", "a person with flushed cheeks", "a face with rosy cheeks"],
    "Sideburns":            ["a person with sideburns", "a person with long sideburns", "a face with sideburns"],
    "Smiling":              ["a smiling person", "a person who is smiling", "a person with a happy expression", "a person with a big smile", "a portrait of a smiling person", "a face with a smile"],
    "Straight_Hair":        ["a person with straight hair", "a person with smooth straight hair", "a portrait of a person with straight hair"],
    "Wavy_Hair":            ["a person with wavy hair", "a person with curly wavy hair", "a portrait of a person with wavy hair"],
    "Wearing_Earrings":     ["a person wearing earrings", "a person with earrings", "a portrait of a person wearing earrings"],
    "Wearing_Hat":          ["a person wearing a hat", "a person with a hat", "a portrait of a person wearing a hat"],
    "Wearing_Lipstick":     ["a person wearing lipstick", "a person with lipstick", "a portrait of a person wearing lipstick"],
    "Wearing_Necklace":     ["a person wearing a necklace", "a person with a necklace", "a portrait of a person wearing a necklace"],
    "Wearing_Necktie":      ["a person wearing a necktie", "a person with a tie", "a portrait of a person wearing a necktie"],
    "Young":                ["a young person", "a youthful person", "a person who looks young", "a portrait of a young person", "a young-looking face"],
}


humanized_mappings_neg: dict[str, list[str]] = {
    "5_o_Clock_Shadow":     ["a clean-shaven person", "a person with no facial stubble", "a person without a 5 o'clock shadow", "a smoothly shaven face"],
    "Arched_Eyebrows":      ["a person with flat eyebrows", "a person whose eyebrows are not arched", "a face with straight eyebrows"],
    "Attractive":           ["an unattractive person", "a plain-looking person", "an ordinary-looking person"],
    "Bags_Under_Eyes":      ["a person without bags under the eyes", "a person with no eye bags", "a fresh-looking face without under-eye bags"],
    "Bald":                 ["a person with hair", "a person with a full head of hair", "a person who is not bald"],
    "Bangs":                ["a person without bangs", "a person with no fringe", "a face without bangs"],
    "Big_Lips":             ["a person with thin lips", "a person with small lips", "a person without big lips"],
    "Big_Nose":             ["a person with a small nose", "a person without a big nose", "a face with a small nose"],
    "Black_Hair":           ["a person without black hair", "a person whose hair is not black"],
    "Blond_Hair":           ["a person without blond hair", "a person whose hair is not blonde"],
    "Blurry":               ["a sharp clear photo of a person", "a high quality in-focus portrait", "a crisp clear image of a face"],
    "Brown_Hair":           ["a person without brown hair", "a person whose hair is not brown"],
    "Bushy_Eyebrows":       ["a person with thin eyebrows", "a person without bushy eyebrows"],
    "Chubby":               ["a thin person", "a person with a slim face", "a person who is not chubby"],
    "Double_Chin":          ["a person without a double chin", "a person with a defined jawline"],
    "Eyeglasses":           ["a person without eyeglasses", "a person not wearing glasses", "a face without glasses", "a person with bare eyes"],
    "Goatee":               ["a person without a goatee", "a clean-shaven person", "a person with no goatee"],
    "Gray_Hair":            ["a person without gray hair", "a person whose hair is not gray"],
    "Heavy_Makeup":         ["a person with no makeup", "a person without makeup", "a face without heavy makeup", "a person with a bare natural face"],
    "High_Cheekbones":      ["a person without high cheekbones", "a person with flat cheeks"],
    "Male":                 ["a woman", "a female person", "a portrait of a woman", "a female face"],
    "Mouth_Slightly_Open":  ["a person with a closed mouth", "a person with closed lips", "a person whose mouth is shut"],
    "Mustache":             ["a clean-shaven person", "a person without a mustache", "a person with no mustache"],
    "Narrow_Eyes":          ["a person with wide eyes", "a person with big eyes", "a person without narrow eyes"],
    "No_Beard":             ["a person with a beard", "a bearded person", "a person with facial hair"],
    "Oval_Face":            ["a person without an oval face", "a person with a round face", "a person with a square face"],
    "Pale_Skin":            ["a person with dark skin", "a person with a tanned complexion", "a person without pale skin"],
    "Pointy_Nose":          ["a person with a rounded nose", "a person without a pointy nose"],
    "Receding_Hairline":    ["a person with a full hairline", "a person without a receding hairline"],
    "Rosy_Cheeks":          ["a person without rosy cheeks", "a person with pale cheeks"],
    "Sideburns":            ["a clean-shaven person", "a person without sideburns"],
    "Smiling":              ["a person with a neutral expression", "a person who is not smiling", "a serious-looking person", "a person with a straight face"],
    "Straight_Hair":        ["a person with curly hair", "a person without straight hair"],
    "Wavy_Hair":            ["a person with straight hair", "a person without wavy hair"],
    "Wearing_Earrings":     ["a person without earrings", "a person not wearing any earrings"],
    "Wearing_Hat":          ["a person without a hat", "a person not wearing a hat", "a bare-headed person"],
    "Wearing_Lipstick":     ["a person without lipstick", "a person with bare lips"],
    "Wearing_Necklace":     ["a person without a necklace", "a person with a bare neck"],
    "Wearing_Necktie":      ["a person without a necktie", "a person with an open collar"],
    "Young":                ["an old person", "an elderly person", "an older person", "a senior person"],
}


clip_imagenet_templates: list[str] = [
    "a bad photo of {phrase}.", "a photo of many {phrase}.", "a sculpture of {phrase}.",
    "a photo of the hard to see {phrase}.", "a low resolution photo of {phrase}.",
    "a rendering of {phrase}.", "graffiti of {phrase}.", "a bad photo of {phrase}.",
    "a cropped photo of {phrase}.", "a tattoo of {phrase}.", "the embroidered {phrase}.",
    "a photo of a hard to see {phrase}.", "a bright photo of {phrase}.", "a photo of a clean {phrase}.",
    "a photo of a dirty {phrase}.", "a dark photo of {phrase}.", "a drawing of {phrase}.",
    "a photo of my {phrase}.", "the plastic {phrase}.", "a photo of the cool {phrase}.",
    "a close-up photo of {phrase}.", "a black and white photo of {phrase}.", "a painting of {phrase}.",
    "a painting of {phrase}.", "a pixelated photo of {phrase}.", "a sculpture of {phrase}.",
    "a bright photo of {phrase}.", "a cropped photo of {phrase}.", "a plastic {phrase}.",
    "a photo of the dirty {phrase}.", "a jpeg corrupted photo of {phrase}.",
    "a blurry photo of {phrase}.", "a photo of {phrase}.", "a good photo of {phrase}.",
    "a rendering of {phrase}.", "a {phrase} in a video game.", "a photo of one {phrase}.",
    "a doodle of {phrase}.", "a close-up photo of {phrase}.", "a photo of {phrase}.",
    "the origami {phrase}.", "a sketch of {phrase}.", "a doodle of {phrase}.",
    "a origami {phrase}.", "a low resolution photo of {phrase}.", "the toy {phrase}.",
    "a rendition of {phrase}.", "a photo of the clean {phrase}.", "a photo of a large {phrase}.",
    "a rendition of {phrase}.", "a photo of a nice {phrase}.", "a photo of a weird {phrase}.",
    "a blurry photo of {phrase}.", "a cartoon {phrase}.", "art of {phrase}.",
    "a sketch of {phrase}.", "a embroidered {phrase}.", "a pixelated photo of {phrase}.",
    "itap of {phrase}.", "a jpeg corrupted photo of {phrase}.", "a good photo of {phrase}.",
    "a plushie {phrase}.", "a photo of the nice {phrase}.", "a photo of the small {phrase}.",
    "a photo of the weird {phrase}.", "the cartoon {phrase}.", "art of {phrase}.",
    "a drawing of {phrase}.", "a photo of the large {phrase}.", "a black and white photo of {phrase}.",
    "the plushie {phrase}.", "a dark photo of {phrase}.", "itap of {phrase}.",
    "graffiti of {phrase}.", "a toy {phrase}.", "itap of {phrase}.",
    "a photo of a cool {phrase}.", "a photo of a small {phrase}.", "a tattoo of {phrase}.",
]


portrait_templates: list[str] = [
    "a portrait of {phrase}.",
    "a portrait photograph of {phrase}.",
    "a closeup headshot of {phrase}.",
    "a candid photo of {phrase}.",
    "a studio portrait of {phrase}.",
    "a high-resolution headshot of {phrase}.",
    "a face photo of {phrase}.",
    "a photo showing the face of {phrase}.",
    "a frontal photo of {phrase}.",
    "a clear photo of {phrase}.",
]


prompt_templates_v2: list[str] = clip_imagenet_templates + portrait_templates
