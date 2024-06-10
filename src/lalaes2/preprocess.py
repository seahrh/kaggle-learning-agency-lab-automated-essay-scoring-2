from typing import AnyStr

from scml import nlp as snlp

__all__ = ["BasicPreprocessor"]


class Preprocessor:
    def __call__(self, s: AnyStr, **kwargs) -> str:
        raise NotImplementedError("Implement this method in subclass")


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def __call__(self, s: AnyStr, **kwargs) -> str:
        res: str = snlp.to_ascii(s)
        res = snlp.collapse_whitespace(res)
        return res
