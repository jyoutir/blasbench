"""Text normalization for Irish ASR evaluation.

Irish (Gaeilge) requires special care during normalization:

- **Fadas** (síneadh fada): á, é, í, ó, ú are distinct phonemes.
  Stripping them conflates different words (céad=hundred vs cead=permission).
  They must ALWAYS be preserved.

- **Lenition** (séimhiú): 'h' inserted after initial consonants in grammatical
  contexts: bean → an bhean. Produces bh, ch, dh, fh, gh, mh, ph, sh, th.

- **Eclipsis** (urú): consonant prefixed in grammatical contexts:
  bád → ar an mbád. Produces mb, gc, nd, bhf, ng, bp, dt, n- (before vowels).

Both mutations are correct orthographic forms under An Caighdeán Oifigiúil
(the official standard) and must be preserved in evaluation.

Unicode NFC normalization is critical: á can be encoded as precomposed (U+00E1)
or decomposed (a + U+0301). Without NFC, visually identical text produces
false WER errors.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Protocol


class TextNormalizer(Protocol):
    """Protocol for text normalizers."""

    def __call__(self, text: str) -> str: ...


class BaseNormalizer(ABC):
    """Abstract base class for text normalizers."""

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Normalize a text string."""
        ...

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a list of texts."""
        return [self(text) for text in texts]


class IrishNormalizer(BaseNormalizer):
    """Irish-specific text normalizer that preserves fadas, lenition, and eclipsis.

    Pipeline:
    1. Unicode NFC normalization (handle decomposed vs precomposed fadas)
    2. Lowercase
    3. Remove punctuation (preserve word characters and whitespace)
    4. Collapse whitespace

    This normalizer NEVER strips diacritics. Fadas are phonemically
    distinctive in Irish and must be preserved for correct WER computation.
    """

    def __call__(self, text: str) -> str:
        # Step 1: NFC normalize — ensures á is a single codepoint, not a + combining accent
        text = unicodedata.normalize("NFC", text)
        # Step 2: Lowercase
        text = text.lower()
        # Step 3: Remove punctuation but preserve word characters (including fadas)
        # \w in Python regex with Unicode matches letters (including accented), digits, underscore
        text = re.sub(r"[^\w\s]", "", text)
        # Step 4: Remove underscores (matched by \w but not desired)
        text = text.replace("_", " ")
        # Step 5: Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class BasicWhisperNormalizer(BaseNormalizer):
    """Wrapper around Whisper's BasicTextNormalizer with remove_diacritics=False.

    Safe for Irish because fadas survive NFKC processing as Unicode
    "Letter" category characters when diacritics are not removed.
    """

    def __init__(self) -> None:
        try:
            from whisper_normalizer.basic import BasicTextNormalizer
        except ImportError:
            # Fall back to transformers' copy
            from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        self._normalizer = BasicTextNormalizer(remove_diacritics=False)

    def __call__(self, text: str) -> str:
        result: str = self._normalizer(text)
        return result


class NoOpNormalizer(BaseNormalizer):
    """Pass-through normalizer that returns text unchanged."""

    def __call__(self, text: str) -> str:
        return text


class CompoundNormalizer(BaseNormalizer):
    """Chains multiple normalizers in sequence."""

    def __init__(self, normalizers: list[BaseNormalizer]) -> None:
        self._normalizers = normalizers

    def __call__(self, text: str) -> str:
        for normalizer in self._normalizers:
            text = normalizer(text)
        return text


def get_normalizer(name: str, remove_diacritics: bool = False) -> BaseNormalizer:
    """Factory function to get a normalizer by name.

    Args:
        name: One of 'irish', 'basic_whisper', 'none'.
        remove_diacritics: Only used for basic_whisper. Must be False for Irish.

    Returns:
        A normalizer instance.

    Raises:
        ValueError: If an unknown normalizer name is provided.
    """
    if name == "irish":
        return IrishNormalizer()
    elif name == "basic_whisper":
        return BasicWhisperNormalizer()
    elif name == "none":
        return NoOpNormalizer()
    else:
        raise ValueError(
            f"Unknown normalizer: {name!r}. Choose from: 'irish', 'basic_whisper', 'none'"
        )
