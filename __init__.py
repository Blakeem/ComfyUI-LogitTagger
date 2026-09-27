from .logit_tagger.nodes import LogitImageCheck, LogitTagger, LogitTextCheck

NODE_CLASS_MAPPINGS = {
    "LogitTagger": LogitTagger,
    "LogitImageCheck": LogitImageCheck,
    "LogitTextCheck": LogitTextCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogitTagger": "Logit Tagger",
    "LogitImageCheck": "Logit Image Check",
    "LogitTextCheck": "Logit Text Check",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
