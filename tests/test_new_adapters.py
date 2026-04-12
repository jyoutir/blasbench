"""Tests for Wav2Vec2, SeamlessM4T, and MMS adapters.

These tests verify adapter construction, properties, and error handling
without loading actual models (which require GPU and network access).
"""

from __future__ import annotations

import numpy as np
import pytest

from blasbench.adapters.mms_adapter import MMSAdapter
from blasbench.adapters.seamless_adapter import SeamlessM4TAdapter
from blasbench.adapters.wav2vec2_adapter import Wav2Vec2Adapter
from blasbench.registry import model_registry

# ─────────────────────────────────────────────────────────────────
# Wav2Vec2Adapter
# ─────────────────────────────────────────────────────────────────


class TestWav2Vec2Adapter:
    def test_defaults(self) -> None:
        adapter = Wav2Vec2Adapter()
        assert adapter.name == "cpierse/wav2vec2-large-xlsr-53-irish"
        assert adapter.num_parameters is None

    def test_custom_model_name(self) -> None:
        adapter = Wav2Vec2Adapter(model_name="some/model")
        assert adapter.name == "some/model"

    def test_transcribe_before_load_raises(self) -> None:
        adapter = Wav2Vec2Adapter()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.transcribe_batch([np.zeros(16000)], [16000])

    def test_unload_without_load_is_safe(self) -> None:
        adapter = Wav2Vec2Adapter()
        adapter.unload()  # Should not raise

    def test_registered_in_registry(self) -> None:
        assert "wav2vec2" in model_registry
        assert model_registry.get("wav2vec2") is Wav2Vec2Adapter

    def test_device_default(self) -> None:
        adapter = Wav2Vec2Adapter(device="cpu")
        assert adapter._device == "cpu"


# ─────────────────────────────────────────────────────────────────
# SeamlessM4TAdapter
# ─────────────────────────────────────────────────────────────────


class TestSeamlessM4TAdapter:
    def test_defaults(self) -> None:
        adapter = SeamlessM4TAdapter()
        assert adapter.name == "facebook/seamless-m4t-v2-large"
        assert adapter.num_parameters is None

    def test_custom_params(self) -> None:
        adapter = SeamlessM4TAdapter(
            model_name="meta/seamless-custom",
            tgt_lang="eng",
            torch_dtype="float32",
        )
        assert adapter.name == "meta/seamless-custom"

    def test_transcribe_before_load_raises(self) -> None:
        adapter = SeamlessM4TAdapter()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.transcribe_batch([np.zeros(16000)], [16000])

    def test_unload_without_load_is_safe(self) -> None:
        adapter = SeamlessM4TAdapter()
        adapter.unload()  # Should not raise

    def test_registered_in_registry(self) -> None:
        assert "seamless" in model_registry
        assert model_registry.get("seamless") is SeamlessM4TAdapter


# ─────────────────────────────────────────────────────────────────
# MMSAdapter
# ─────────────────────────────────────────────────────────────────


class TestMMSAdapter:
    def test_defaults(self) -> None:
        adapter = MMSAdapter()
        assert adapter.name == "facebook/mms-1b-all"
        assert adapter.num_parameters is None

    def test_custom_params(self) -> None:
        adapter = MMSAdapter(model_name="meta/mms-custom", target_lang="eng")
        assert adapter.name == "meta/mms-custom"

    def test_transcribe_before_load_raises(self) -> None:
        adapter = MMSAdapter()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.transcribe_batch([np.zeros(16000)], [16000])

    def test_unload_without_load_is_safe(self) -> None:
        adapter = MMSAdapter()
        adapter.unload()  # Should not raise

    def test_registered_in_registry(self) -> None:
        assert "mms" in model_registry
        assert model_registry.get("mms") is MMSAdapter

    def test_target_lang_default(self) -> None:
        adapter = MMSAdapter()
        assert adapter._target_lang == "gle"
