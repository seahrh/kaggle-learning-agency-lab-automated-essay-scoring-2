from typing import Set

import scml
import spacy
from scml import nlp as snlp

__all__ = [
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "repeat_char_frac",
    "repeat_substring_frac",
    "unique_word_frac",
    "StopwordFraction",
]

log = scml.get_logger(__name__)


def digit_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return snlp.count_digit(s) / len(s)  # type: ignore


def letter_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return snlp.count_alpha(s) / len(s)  # type: ignore


def space_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return snlp.count_space(s) / len(s)  # type: ignore


def punc_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return snlp.count_punctuation(s) / len(s)  # type: ignore


def upper_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return snlp.count_upper(s) / len(s)  # type: ignore


_rc = snlp.RepeatingCharacter(max_times=3, letters=True, punctuation=True)
_rs = snlp.RepeatingSubstring(min_length=3, max_times=1, letters=True, punctuation=True)


def repeat_char_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return _rc.count(s) / len(s)  # type: ignore


def repeat_substring_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    return _rs.count_char(s) / len(s)  # type: ignore


def unique_word_frac(s: str) -> float:
    if len(s) == 0:
        return 0
    words = s.split()
    return len(set(words)) / len(words)  # type: ignore


class StopwordFraction:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg", exclude=["textcat", "ner", "tok2vec"])
        log.debug(self.nlp.pipe_names)
        self.stops: Set[str] = self.nlp.Defaults.stop_words

    def __call__(self, s: str, *args, **kwargs) -> float:
        if len(s) == 0:
            return 0
        words = s.split()
        n = 0
        for word in words:
            if word in self.stops:
                n += 1
        return n / len(words)  # type: ignore
