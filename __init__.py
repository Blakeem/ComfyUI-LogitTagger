from .logit_tagger.node import LogitTagger

NODE_CLASS_MAPPINGS = {
    "LogitTagger": LogitTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogitTagger": "Logit Tagger",
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
