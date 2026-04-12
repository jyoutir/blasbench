"""Tests for Irish text normalizer.

These tests verify that Irish linguistic features (fadas, lenition, eclipsis)
are correctly preserved during normalization, while punctuation and case
are correctly removed/lowered.
"""

from __future__ import annotations

import pytest

from blasbench.normalizer import (
    CompoundNormalizer,
    IrishNormalizer,
    NoOpNormalizer,
    get_normalizer,
)


class TestIrishNormalizer:
    """Test IrishNormalizer preserves Irish linguistic features."""

    @pytest.fixture()
    def norm(self) -> IrishNormalizer:
        return IrishNormalizer()

    # --- Fada preservation ---

    def test_preserves_fadas(self, norm: IrishNormalizer) -> None:
        """Fadas (á, é, í, ó, ú) must be preserved — they distinguish words."""
        assert norm("Céad míle fáilte!") == "céad míle fáilte"

    def test_fada_distinguishes_words(self, norm: IrishNormalizer) -> None:
        """céad (hundred) ≠ cead (permission) — fada is phonemically distinctive."""
        assert norm("céad") != norm("cead")
        assert norm("céad") == "céad"
        assert norm("cead") == "cead"

    def test_all_fada_vowels(self, norm: IrishNormalizer) -> None:
        assert norm("á é í ó ú") == "á é í ó ú"
        assert norm("Á É Í Ó Ú") == "á é í ó ú"

    # --- Lenition preservation ---

    def test_preserves_lenition(self, norm: IrishNormalizer) -> None:
        """Lenition (séimhiú): 'h' after initial consonant is grammatically correct."""
        assert norm("An bhean") == "an bhean"
        assert norm("mo mháthair") == "mo mháthair"
        assert norm("a chara") == "a chara"

    def test_all_lenited_consonants(self, norm: IrishNormalizer) -> None:
        """All nine lenitable consonants: b, c, d, f, g, m, p, s, t."""
        pairs = [
            ("bh", "bhean"),
            ("ch", "chara"),
            ("dh", "dhuit"),
            ("fh", "fhear"),
            ("gh", "ghrian"),
            ("mh", "mháthair"),
            ("ph", "pháiste"),
            ("sh", "sholas"),
            ("th", "thír"),
        ]
        for _, word in pairs:
            assert word in norm(word)

    # --- Eclipsis preservation ---

    def test_preserves_eclipsis(self, norm: IrishNormalizer) -> None:
        """Eclipsis (urú): consonant prefix is grammatically correct."""
        assert norm("ar an mbád") == "ar an mbád"
        assert norm("i nGaillimh") == "i ngaillimh"

    def test_eclipsis_patterns(self, norm: IrishNormalizer) -> None:
        """All eclipsis patterns: mb, gc, nd, bhf, ng, bp, dt."""
        assert "mb" in norm("mbád")
        assert "gc" in norm("gcathaoir")
        assert "nd" in norm("ndoras")
        assert "bhf" in norm("bhfuil")
        assert "ng" in norm("ngairdín")
        assert "bp" in norm("bpáiste")
        assert "dt" in norm("dteach")

    # --- Punctuation removal ---

    def test_removes_punctuation(self, norm: IrishNormalizer) -> None:
        assert norm("Dia duit, a chara!") == "dia duit a chara"
        assert norm("Cén t-ainm atá ort?") == "cén tainm atá ort"

    def test_removes_quotes_and_brackets(self, norm: IrishNormalizer) -> None:
        assert norm('"Tá sé go maith"') == "tá sé go maith"
        assert norm("(nó rud éigin)") == "nó rud éigin"

    # --- Unicode normalization ---

    def test_nfc_normalization(self, norm: IrishNormalizer) -> None:
        """Decomposed á (a + combining accent) must equal precomposed á."""
        precomposed = "á"  # U+00E1
        decomposed = "a\u0301"  # a + combining acute
        assert norm(precomposed) == norm(decomposed)
        assert norm(decomposed) == "á"

    # --- Whitespace ---

    def test_collapses_whitespace(self, norm: IrishNormalizer) -> None:
        assert norm("  tá   sé   ") == "tá sé"

    def test_empty_string(self, norm: IrishNormalizer) -> None:
        assert norm("") == ""

    # --- Numbers and digits ---

    def test_preserves_digits(self, norm: IrishNormalizer) -> None:
        assert norm("3 bliana") == "3 bliana"


class TestNoOpNormalizer:
    def test_passthrough(self) -> None:
        norm = NoOpNormalizer()
        assert norm("Hello, World!") == "Hello, World!"


class TestCompoundNormalizer:
    def test_chains_normalizers(self) -> None:
        compound = CompoundNormalizer([IrishNormalizer(), NoOpNormalizer()])
        assert compound("Céad Míle Fáilte!") == "céad míle fáilte"


class TestGetNormalizer:
    def test_irish(self) -> None:
        norm = get_normalizer("irish")
        assert isinstance(norm, IrishNormalizer)

    def test_none(self) -> None:
        norm = get_normalizer("none")
        assert isinstance(norm, NoOpNormalizer)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalizer"):
            get_normalizer("english")


class TestNormalizeBatch:
    def test_batch(self) -> None:
        norm = IrishNormalizer()
        results = norm.normalize_batch(["Tá", "Níl"])
        assert results == ["tá", "níl"]
