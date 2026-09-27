from .logit_tagger.nodes import LogitImageCheck, LogitImageClassify, LogitPromptCheck, LogitTagger, LogitTextCheck

NODE_CLASS_MAPPINGS = {
    "LogitTagger": LogitTagger,
    "LogitImageCheck": LogitImageCheck,
    "LogitTextCheck": LogitTextCheck,
    "LogitPromptCheck": LogitPromptCheck,
    "LogitImageClassify": LogitImageClassify,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogitTagger": "Logit Tagger",
    "LogitImageCheck": "Logit Image Check",
    "LogitTextCheck": "Logit Text Check",
    "LogitPromptCheck": "Logit Prompt Check",
    "LogitImageClassify": "Logit Image Classify",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
