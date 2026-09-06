"""
tests/test_structuring_engine.py
----------------------------------
Unit tests for kiosk_agent/structuring_engine.py.

Coverage targets:
  - Stub mode returns valid StructuredComplaint for all 3 MVP languages
  - Stub description is truncated to ≤280 chars (schema constraint)
  - Stub location_hint derived from transcript (truncated to 60 chars)
  - StructuringModelNotReadyError raised before model is loaded
  - StructuringTimeoutError raised when generation exceeds timeout
  - StructuringOutputInvalidError raised when JSON parse fails
  - StructuringOutputInvalidError raised when Pydantic validation fails
  - Normal inference path: mock llm returns valid JSON → valid StructuredComplaint
  - All 6 category values accepted; any 7th raises ValidationError
  - Prompt template loading: language-specific file loaded, fallback on missing
  - load_model() is a no-op in stub mode
  - load_model() raises FileNotFoundError when GBNF file missing
  - unload_model() resets all global state
  - Schema validation: urgency_keywords capped at 5 elements
  - Schema validation: structuring_confidence must be in [0.0, 1.0]

All tests run without the GGUF binary or llama-cpp-python installed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from kiosk_agent.schemas import Category, StructuredComplaint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset module-level _llm, _grammar, _grammar_str, _PROMPT_CACHE before each test."""
    import kiosk_agent.structuring_engine as eng
    eng._llm = None
    eng._grammar = None
    eng._grammar_str = ""
    eng._PROMPT_CACHE.clear()
    yield
    eng._llm = None
    eng._grammar = None
    eng._grammar_str = ""
    eng._PROMPT_CACHE.clear()


@pytest.fixture()
def stub_on(monkeypatch):
    monkeypatch.setattr("kiosk_agent.structuring_engine.settings.stub_structuring", True)
    monkeypatch.setattr(
        "kiosk_agent.structuring_engine.settings.structuring_timeout_s", 20
    )


@pytest.fixture()
def stub_off(monkeypatch):
    monkeypatch.setattr("kiosk_agent.structuring_engine.settings.stub_structuring", False)
    monkeypatch.setattr(
        "kiosk_agent.structuring_engine.settings.structuring_timeout_s", 20
    )


def _valid_json_output(
    category: str = "ROAD",
    confidence: float = 0.91,
    n_keywords: int = 3,
) -> str:
    """Build a valid Structured Complaint JSON string for use in mock llm output."""
    data = {
        "category": category,
        "subcategory": "Pothole",
        "description": "Large pothole on main road causing vehicle damage.",
        "location_hint": "near community centre",
        "reported_asset_type": "road_segment",
        "urgency_keywords": ["pothole", "damage", "road"][:n_keywords],
        "structuring_confidence": confidence,
    }
    return json.dumps(data)


def _install_mock_llm(monkeypatch, generated_text: str):
    """
    Install a mock _llm that returns `generated_text` as its generation output.
    Replicates the llama_cpp.Llama.__call__ return shape.
    """
    import kiosk_agent.structuring_engine as eng

    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [{"text": generated_text}]
    }
    monkeypatch.setattr(eng, "_llm", mock_llm)
    monkeypatch.setattr(eng, "_grammar", MagicMock())
    return mock_llm


# ---------------------------------------------------------------------------
# Stub mode tests
# ---------------------------------------------------------------------------

class TestStubMode:
    @pytest.mark.parametrize("lang", ["hi", "mr", "ta"])
    def test_stub_returns_valid_schema_for_all_languages(self, stub_on, lang):
        from kiosk_agent.structuring_engine import structure
        result = structure("dummy transcript", lang)
        assert isinstance(result, StructuredComplaint)
        assert isinstance(result.category, Category)
        assert 0.0 <= result.structuring_confidence <= 1.0
        assert len(result.description) <= 280
        assert len(result.urgency_keywords) <= 5

    def test_stub_description_within_schema_limit(self, stub_on):
        """Stub description must not exceed 280 chars (architecture.md §6.1)."""
        from kiosk_agent.structuring_engine import structure
        result = structure("x" * 300, "hi")
        assert len(result.description) <= 280

    def test_stub_location_hint_derived_from_transcript(self, stub_on):
        """location_hint must be the first 60 chars of the transcript."""
        from kiosk_agent.structuring_engine import structure
        transcript = "A" * 80
        result = structure(transcript, "hi")
        assert result.location_hint == "A" * 60

    def test_stub_empty_transcript_fallback(self, stub_on):
        """Empty transcript must not crash — falls back to default location_hint."""
        from kiosk_agent.structuring_engine import structure
        result = structure("", "hi")
        assert len(result.location_hint) > 0

    def test_load_model_noop_in_stub_mode(self, stub_on):
        import kiosk_agent.structuring_engine as eng
        eng.load_model()
        assert eng._llm is None

    def test_is_ready_true_in_stub_mode(self, stub_on):
        from kiosk_agent.structuring_engine import is_ready
        assert is_ready() is True


# ---------------------------------------------------------------------------
# Model-not-ready guard
# ---------------------------------------------------------------------------

class TestModelNotReady:
    def test_structure_raises_if_model_not_loaded(self, stub_off):
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringModelNotReadyError,
        )
        with pytest.raises(StructuringModelNotReadyError):
            structure("some transcript", "hi")

    def test_is_ready_false_before_load(self, stub_off):
        from kiosk_agent.structuring_engine import is_ready
        assert is_ready() is False

    def test_is_ready_true_after_mock_load(self, stub_off, monkeypatch):
        import kiosk_agent.structuring_engine as eng
        monkeypatch.setattr(eng, "_llm", MagicMock())
        assert eng.is_ready() is True


# ---------------------------------------------------------------------------
# Normal inference path (mocked llm)
# ---------------------------------------------------------------------------

class TestNormalInference:
    def test_valid_json_produces_structured_complaint(self, stub_off, monkeypatch):
        from kiosk_agent.structuring_engine import structure
        _install_mock_llm(monkeypatch, _valid_json_output())
        result = structure("Large pothole on the main road.", "hi")
        assert isinstance(result, StructuredComplaint)
        assert result.category == Category.ROAD
        assert result.subcategory == "Pothole"
        assert result.structuring_confidence == pytest.approx(0.91)

    @pytest.mark.parametrize("category", [
        "ROAD", "WATER", "ELECTRICITY", "SANITATION", "STREETLIGHT", "OTHER"
    ])
    def test_all_six_categories_accepted(self, stub_off, monkeypatch, category):
        """All 6 category enum values must produce a valid StructuredComplaint."""
        from kiosk_agent.structuring_engine import structure
        _install_mock_llm(monkeypatch, _valid_json_output(category=category))
        result = structure("test transcript", "hi")
        assert result.category.value == category

    def test_temperature_is_zero_deterministic(self, stub_off, monkeypatch):
        """
        temperature=0.0 must be passed to the mock llm call.
        ADR-002: no randomness in constrained decoding.
        """
        from kiosk_agent.structuring_engine import structure
        mock_llm = _install_mock_llm(monkeypatch, _valid_json_output())
        structure("test", "hi")
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0

    def test_grammar_passed_to_llm(self, stub_off, monkeypatch):
        """The GBNF grammar instance must be passed as the `grammar=` kwarg."""
        import kiosk_agent.structuring_engine as eng
        from kiosk_agent.structuring_engine import structure
        mock_llm = _install_mock_llm(monkeypatch, _valid_json_output())
        grammar_sentinel = eng._grammar
        structure("test", "hi")
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs.get("grammar") is grammar_sentinel

    def test_trailing_brace_appended_if_missing(self, stub_off, monkeypatch):
        """
        If the model stops generation without the closing brace (stop token triggered
        mid-brace), the engine must append '}' before JSON parsing.
        """
        from kiosk_agent.structuring_engine import structure
        # Strip the closing brace from valid output.
        incomplete = _valid_json_output().rstrip("}")
        _install_mock_llm(monkeypatch, incomplete)
        # Should not raise — the engine appends the missing brace.
        result = structure("test", "hi")
        assert isinstance(result, StructuredComplaint)


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_raises_structuring_timeout_error(self, stub_off, monkeypatch):
        """
        When the LLM thread does not finish within structuring_timeout_s,
        StructuringTimeoutError must be raised.
        implementation.md §4.3: "Raises StructuringTimeoutError if generation
        exceeds 20 seconds."
        """
        import kiosk_agent.structuring_engine as eng
        from kiosk_agent.structuring_engine import StructuringTimeoutError

        # Install a mock llm that sleeps longer than the timeout.
        def _slow_llm(*args, **kwargs):
            time.sleep(5)  # simulate slow generation
            return {"choices": [{"text": _valid_json_output()}]}

        mock_llm = MagicMock(side_effect=_slow_llm)
        monkeypatch.setattr(eng, "_llm", mock_llm)
        monkeypatch.setattr(eng, "_grammar", MagicMock())
        # Set a very short timeout so the test completes quickly.
        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.structuring_timeout_s", 1
        )

        with pytest.raises(StructuringTimeoutError) as exc_info:
            eng.structure("test transcript", "hi")

        assert exc_info.value.timeout_s == 1

    def test_timeout_value_from_settings(self, stub_off, monkeypatch):
        """StructuringTimeoutError.timeout_s must match settings.structuring_timeout_s."""
        import kiosk_agent.structuring_engine as eng
        from kiosk_agent.structuring_engine import StructuringTimeoutError

        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.structuring_timeout_s", 2
        )

        def _slow(*args, **kwargs):
            time.sleep(10)

        monkeypatch.setattr(eng, "_llm", MagicMock(side_effect=_slow))
        monkeypatch.setattr(eng, "_grammar", MagicMock())

        with pytest.raises(StructuringTimeoutError) as exc_info:
            eng.structure("test", "hi")

        assert exc_info.value.timeout_s == 2


# ---------------------------------------------------------------------------
# Invalid / adversarial output tests
# ---------------------------------------------------------------------------

class TestInvalidOutput:
    def test_bad_json_raises_output_invalid_error(self, stub_off, monkeypatch):
        """Malformed JSON from llm must raise StructuringOutputInvalidError."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        _install_mock_llm(monkeypatch, "{bad json!!!}")
        with pytest.raises(StructuringOutputInvalidError, match="not valid JSON"):
            structure("test", "hi")

    def test_wrong_category_raises_output_invalid_error(self, stub_off, monkeypatch):
        """A category value not in the enum must fail Pydantic validation."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        # "BRIDGE" is not a valid category — this should be unreachable via GBNF
        # but we test the belt-and-suspenders Pydantic layer.
        bad_output = _valid_json_output().replace('"ROAD"', '"BRIDGE"')
        _install_mock_llm(monkeypatch, bad_output)
        with pytest.raises(StructuringOutputInvalidError):
            structure("test", "hi")

    def test_too_many_keywords_raises_output_invalid_error(self, stub_off, monkeypatch):
        """urgency_keywords with 6+ elements must fail StructuredComplaint validation."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        data = json.loads(_valid_json_output())
        data["urgency_keywords"] = ["a", "b", "c", "d", "e", "f"]  # 6 — exceeds max 5
        _install_mock_llm(monkeypatch, json.dumps(data))
        with pytest.raises(StructuringOutputInvalidError):
            structure("test", "hi")

    def test_confidence_out_of_range_raises_output_invalid_error(
        self, stub_off, monkeypatch
    ):
        """structuring_confidence > 1.0 must fail StructuredComplaint validation."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        data = json.loads(_valid_json_output())
        data["structuring_confidence"] = 1.5  # invalid
        _install_mock_llm(monkeypatch, json.dumps(data))
        with pytest.raises(StructuringOutputInvalidError):
            structure("test", "hi")

    def test_missing_required_field_raises_output_invalid_error(
        self, stub_off, monkeypatch
    ):
        """JSON missing a required field must fail StructuredComplaint validation."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        data = json.loads(_valid_json_output())
        del data["description"]  # required field removed
        _install_mock_llm(monkeypatch, json.dumps(data))
        with pytest.raises(StructuringOutputInvalidError):
            structure("test", "hi")

    def test_empty_output_raises_structuring_error(self, stub_off, monkeypatch):
        """Empty string output from llm must raise StructuringError."""
        from kiosk_agent.structuring_engine import (
            structure,
            StructuringOutputInvalidError,
        )
        _install_mock_llm(monkeypatch, "")
        with pytest.raises(StructuringOutputInvalidError):
            structure("test", "hi")


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    def test_language_specific_template_loaded(
        self, stub_off, monkeypatch, tmp_path
    ):
        """
        When a language-specific template file exists, it must be used
        (not the default English template).
        """
        import kiosk_agent.structuring_engine as eng

        # Create a temp prompts dir with a Hindi template.
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        hi_template = prompts_dir / "hi.txt"
        hi_template.write_text("Hindi prompt: {transcript}\nJSON:", encoding="utf-8")

        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.prompts_dir", prompts_dir
        )

        template = eng._load_prompt_template("hi")
        assert "Hindi prompt" in template
        assert "{transcript}" in template

    def test_missing_template_falls_back_to_default(
        self, stub_off, monkeypatch, tmp_path
    ):
        """Missing language template must use the default English template."""
        import kiosk_agent.structuring_engine as eng

        empty_dir = tmp_path / "empty_prompts"
        empty_dir.mkdir()
        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.prompts_dir", empty_dir
        )

        template = eng._load_prompt_template("xx")  # unsupported language
        assert "{transcript}" in template
        assert len(template) > 20  # non-empty fallback

    def test_template_cached_after_first_load(
        self, stub_off, monkeypatch, tmp_path
    ):
        """_load_prompt_template must cache results (not re-read disk on every call)."""
        import kiosk_agent.structuring_engine as eng

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "hi.txt").write_text("Template: {transcript}", encoding="utf-8")

        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.prompts_dir", prompts_dir
        )

        t1 = eng._load_prompt_template("hi")
        t2 = eng._load_prompt_template("hi")
        assert t1 is t2  # same object — cached

    def test_transcript_interpolated_into_prompt(
        self, stub_off, monkeypatch
    ):
        """The transcript must appear verbatim inside the prompt sent to the llm."""
        import kiosk_agent.structuring_engine as eng
        from kiosk_agent.structuring_engine import structure

        mock_llm = _install_mock_llm(monkeypatch, _valid_json_output())

        # Override prompt template to a simple known form.
        monkeypatch.setattr(
            eng, "_PROMPT_CACHE", {"hi": "PROMPT: {transcript} END"}
        )

        transcript = "सड़क पर गड्ढा है।"
        structure(transcript, "hi")

        prompt_sent = mock_llm.call_args.args[0]
        assert transcript in prompt_sent
        assert "PROMPT:" in prompt_sent


# ---------------------------------------------------------------------------
# Model lifecycle tests
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    def test_load_model_raises_if_grammar_file_missing(
        self, stub_off, monkeypatch, tmp_path
    ):
        import kiosk_agent.structuring_engine as eng

        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.grammar_path",
            tmp_path / "nonexistent.gbnf",
        )
        with pytest.raises(FileNotFoundError):
            eng.load_model()
        assert eng._llm is None

    def test_load_model_raises_if_gguf_missing(
        self, stub_off, monkeypatch, tmp_path
    ):
        import kiosk_agent.structuring_engine as eng

        # Create a real (but minimal) grammar file so the grammar load passes.
        grammar_file = tmp_path / "grammar.gbnf"
        grammar_file.write_text('root ::= "test"', encoding="utf-8")
        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.grammar_path", grammar_file
        )
        monkeypatch.setattr(
            "kiosk_agent.structuring_engine.settings.llm_model_path",
            tmp_path / "nonexistent.gguf",
        )

        with pytest.raises(FileNotFoundError):
            eng.load_model()

        assert eng._llm is None

    def test_load_model_noop_if_already_loaded(self, stub_off, monkeypatch):
        import kiosk_agent.structuring_engine as eng

        sentinel = MagicMock()
        monkeypatch.setattr(eng, "_llm", sentinel)

        eng.load_model()
        assert eng._llm is sentinel

    def test_unload_model_clears_all_state(self, stub_off, monkeypatch):
        import kiosk_agent.structuring_engine as eng

        monkeypatch.setattr(eng, "_llm", MagicMock())
        monkeypatch.setattr(eng, "_grammar", MagicMock())
        monkeypatch.setattr(eng, "_grammar_str", "root ::= test")

        eng.unload_model()

        assert eng._llm is None
        assert eng._grammar is None
        assert eng._grammar_str == ""
