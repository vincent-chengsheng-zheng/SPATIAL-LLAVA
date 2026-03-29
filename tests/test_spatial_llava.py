"""
tests/test_spatial_llava.py

Unit tests for core/model/spatial_llava.py

Strategy:
    SpatialLLaVA loads a 7B model on init — too heavy for dev container.
    We mock the heavy dependencies (LlavaForConditionalGeneration, AutoProcessor)
    and test the real SpatialLLaVA logic: __init__, forward, freeze, save/load.

Run with:
    pytest tests/test_spatial_llava.py -v
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from core.model.spatial_llava import SpatialLLaVA
from core.model.regression_head import RegressionHead


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_llava(hidden_size: int = 64):
    """
    Build a minimal mock LLaVA model that mimics the API
    used by SpatialLLaVA without loading any real weights.
    """
    mock = MagicMock()
    mock.config.text_config.hidden_size = hidden_size

    # vision_tower and language_model need real parameter lists
    # so _freeze_* methods can iterate over them
    vision_layer = nn.Linear(8, 8)
    llm_layer = nn.Linear(8, 8)
    mock.vision_tower = vision_layer
    mock.language_model = llm_layer

    # forward output: last hidden state at correct size
    def fake_forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        seq_len = kwargs["input_ids"].shape[1]
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        out = MagicMock()
        out.hidden_states = [hidden] * 3   # simulate 3 layers
        return out

    mock.side_effect = fake_forward

    # get_peft_model wraps the mock — make it still callable
    mock.__call__ = fake_forward
    return mock


def make_mock_processor(loc_token_id: int = 42):
    """Build a minimal mock processor with a tokenizer."""
    mock = MagicMock()

    def convert(token):
        if token in ("<loc>", "[LOC]", "<LOC>"):
            return loc_token_id
        return 0   # unk

    mock.tokenizer.convert_tokens_to_ids.side_effect = convert
    mock.tokenizer.unk_token_id = 0
    return mock


# ── Fixtures ──────────────────────────────────────────────────────────────────

LOC_TOKEN_ID = 9999
HIDDEN_SIZE = 64   # small for fast tests


@pytest.fixture
def model():
    """
    SpatialLLaVA instance with all heavy deps mocked.
    Uses real RegressionHead and real freeze/forward logic.
    """
    mock_llava = make_mock_llava(hidden_size=HIDDEN_SIZE)
    mock_processor = make_mock_processor(loc_token_id=LOC_TOKEN_ID)

    with patch(
        "core.model.spatial_llava.LlavaForConditionalGeneration.from_pretrained",
        return_value=mock_llava,
    ), patch(
        "core.model.spatial_llava.AutoProcessor.from_pretrained",
        return_value=mock_processor,
    ), patch(
        "core.model.spatial_llava.get_peft_model",
        side_effect=lambda m, cfg: m,   # return model unchanged
    ):
        m = SpatialLLaVA(
            model_id="mock/llava",
            lora_rank=4,
        )
        # Override hidden size to match our small mock
        m.regression_head = RegressionHead(
            hidden_size=HIDDEN_SIZE, intermediate=32
        )
    return m


@pytest.fixture
def dummy_inputs():
    """Batch of 2 samples with [LOC] token at position 5."""
    batch_size = 2
    seq_len = 10
    images = torch.randn(batch_size, 3, 384, 384)
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    input_ids[:, 5] = LOC_TOKEN_ID   # place [LOC] at position 5
    return images, input_ids


# ── Initialization tests ──────────────────────────────────────────────────────

class TestInit:

    def test_is_nn_module(self, model):
        assert isinstance(model, nn.Module)

    def test_has_regression_head(self, model):
        assert isinstance(model.regression_head, RegressionHead)

    def test_has_correct_loc_token_id(self, model):
        assert model.loc_token_id == LOC_TOKEN_ID

    def test_has_processor(self, model):
        assert model.processor is not None

    def test_has_llava(self, model):
        assert model.llava is not None


# ── Freeze tests ──────────────────────────────────────────────────────────────

class TestFreeze:

    def test_freeze_vision_encoder(self, model):
        """All vision tower params should have requires_grad=False after freeze."""
        # Ensure unfrozen first
        for p in model.llava.vision_tower.parameters():
            p.requires_grad = True

        model._freeze_vision_encoder()

        for p in model.llava.vision_tower.parameters():
            assert not p.requires_grad, \
                "Vision encoder params should be frozen"

    def test_freeze_llm_backbone(self, model):
        """All LLM backbone params should have requires_grad=False after freeze."""
        for p in model.llava.language_model.parameters():
            p.requires_grad = True

        model._freeze_llm_backbone()

        for p in model.llava.language_model.parameters():
            assert not p.requires_grad, \
                "LLM backbone params should be frozen"


# ── Forward pass tests ────────────────────────────────────────────────────────

class TestForward:

    def test_output_has_bbox_key(self, model, dummy_inputs):
        images, input_ids = dummy_inputs
        result = model(images, input_ids)
        assert "bbox" in result

    def test_bbox_shape(self, model, dummy_inputs):
        images, input_ids = dummy_inputs
        result = model(images, input_ids)
        assert result["bbox"].shape == (2, 4), \
            f"Expected (2, 4), got {result['bbox'].shape}"

    def test_bbox_values_in_range(self, model, dummy_inputs):
        images, input_ids = dummy_inputs
        result = model(images, input_ids)
        assert (result["bbox"] >= 0).all() and (result["bbox"] <= 1).all(), \
            "All bbox values must be in [0, 1]"

    def test_with_attention_mask(self, model, dummy_inputs):
        images, input_ids = dummy_inputs
        attention_mask = torch.ones_like(input_ids)
        result = model(images, input_ids, attention_mask=attention_mask)
        assert "bbox" in result
        assert result["bbox"].shape == (2, 4)

    def test_single_sample(self, model):
        images = torch.randn(1, 3, 384, 384)
        input_ids = torch.randint(1, 100, (1, 10))
        input_ids[0, 5] = LOC_TOKEN_ID
        result = model(images, input_ids)
        assert result["bbox"].shape == (1, 4)


# ── Parameter counting tests ──────────────────────────────────────────────────

class TestParameterCount:

    def test_returns_dict_with_correct_keys(self, model):
        params = model.count_parameters()
        assert set(params.keys()) == {"total", "trainable", "trainable_percent"}

    def test_trainable_percent_in_range(self, model):
        params = model.count_parameters()
        assert 0 <= params["trainable_percent"] <= 100

    def test_trainable_less_than_total(self, model):
        params = model.count_parameters()
        assert params["trainable"] <= params["total"]

    def test_regression_head_params_are_trainable(self, model):
        """Regression head must always be trainable."""
        head_params = sum(
            p.numel() for p in model.regression_head.parameters()
            if p.requires_grad
        )
        assert head_params > 0, "Regression head should have trainable params"


# ── Save / Load tests ─────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_save_creates_regression_head_file(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "regression_head.pth")), \
                "regression_head.pth should be saved"

    def test_load_restores_regression_head_weights(self, model):
        """Saved and reloaded regression head should have identical weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save_pretrained(tmpdir)

            # Corrupt weights
            with torch.no_grad():
                for p in model.regression_head.parameters():
                    p.fill_(9.99)

            # Reload regression head only (LLaVA part is mocked)
            model.regression_head.load_state_dict(
                torch.load(os.path.join(tmpdir, "regression_head.pth"),
                           map_location="cpu")
            )

            # Verify weights are restored (not 9.99)
            for p in model.regression_head.parameters():
                assert not torch.all(p == 9.99), \
                    "Weights should be restored after load"
