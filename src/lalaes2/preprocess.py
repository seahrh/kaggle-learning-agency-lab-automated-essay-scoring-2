from typing import AnyStr, List, Set

import scml
import spacy
from scml import nlp as snlp

__all__ = ["BasicPreprocessor", "BowPreprocessor"]

log = scml.get_logger(__name__)


class Preprocessor:
    def __call__(self, s: AnyStr, *args, **kwargs) -> str:
        raise NotImplementedError("Implement this method in subclass")


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def __call__(self, s: AnyStr, *args, **kwargs) -> str:
        res: str = snlp.to_ascii(s)
        res = snlp.collapse_whitespace(res)
        return res


class BowPreprocessor(BasicPreprocessor):
    def __init__(self, drop_stopword: bool = False):
        super().__init__()
        self.drop_stopword = drop_stopword
        self.nlp = spacy.load("en_core_web_lg", exclude=["textcat", "ner", "tok2vec"])
        log.debug(self.nlp.pipe_names)
        self.stops: Set[str] = self.nlp.Defaults.stop_words
        log.debug(f"{len(self.stops)} stopwords\n{sorted(list(self.stops))}")
        self.contraction = snlp.ContractionExpansion()

    def __call__(self, s: AnyStr, *args, **kwargs) -> str:
        res: str = super().__call__(s, args, kwargs)
        res = self.contraction.apply(res)
        # Expand contractions before remove punctuation
        res = snlp.strip_punctuation(res, replacement=" ")
        doc = self.nlp(res)
        tokens: List[str] = []
        for token in doc:
            t = token.lemma_.lower()
            if self.drop_stopword and t in self.stops:
                continue
            tokens.append(t)
        res = " ".join(tokens)
        res = snlp.collapse_whitespace(res)
        return res
