from .logit_tagger.nodes import (
    LogitCustomTagger,
    LogitImageCheck,
    LogitImageClassify,
    LogitPromptCheck,
    LogitStyleExtractor,
    LogitTagCheck,
    LogitTagger,
    LogitTextCheck,
    LogitTextClassify,
)

NODE_CLASS_MAPPINGS = {
    "LogitTagger": LogitTagger,
    "LogitCustomTagger": LogitCustomTagger,
    "LogitTagCheck": LogitTagCheck,
    "LogitImageCheck": LogitImageCheck,
    "LogitTextCheck": LogitTextCheck,
    "LogitPromptCheck": LogitPromptCheck,
    "LogitImageClassify": LogitImageClassify,
    "LogitTextClassify": LogitTextClassify,
    "LogitStyleExtractor": LogitStyleExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogitTagger": "Logit Tagger",
    "LogitCustomTagger": "Logit Custom Tagger",
    "LogitTagCheck": "Logit Tag Check",
    "LogitImageCheck": "Logit Image Check",
    "LogitTextCheck": "Logit Text Check",
    "LogitPromptCheck": "Logit Prompt Check",
    "LogitImageClassify": "Logit Image Classify",
    "LogitTextClassify": "Logit Text Classify",
    "LogitStyleExtractor": "Logit Style Extractor",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
