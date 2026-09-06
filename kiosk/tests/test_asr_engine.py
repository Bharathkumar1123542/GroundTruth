"""
tests/test_asr_engine.py
-------------------------
Unit tests for kiosk_agent/asr_engine.py.

Coverage targets:
  - Stub mode returns valid AsrResult for all 3 MVP languages
  - AsrInputTooLongError raised before inference when audio exceeds max duration
  - AsrLowConfidenceError raised when computed confidence < threshold
  - AsrModelNotReadyError raised when transcribe() called without load_model()
  - _pcm_bytes_to_duration() returns correct duration for known PCM buffer
  - _pcm_bytes_to_float32() returns correct sample range [-1, 1]
  - load_model() is a no-op in stub mode
  - Normal inference path with mocked pywhispercpp Model

All tests run without the GGUF model binary or pywhispercpp installed.
The real Model is mocked via monkeypatch / unittest.mock.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_engine():
    """
    Reset the module-level _model and _model_path globals before each test
    to prevent state leakage between tests.
    """
    import kiosk_agent.asr_engine as eng
    eng._model = None
    eng._model_path = ""
    yield
    eng._model = None
    eng._model_path = ""


@pytest.fixture()
def stub_settings(monkeypatch):
    """Force stub mode on for the duration of the test."""
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.stub_structuring", True)
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.asr_confidence_threshold", 0.55)
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.asr_max_duration_s", 90)


@pytest.fixture()
def real_settings(monkeypatch):
    """Force stub mode OFF for tests that exercise the real inference path."""
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.stub_structuring", False)
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.asr_confidence_threshold", 0.55)
    monkeypatch.setattr("kiosk_agent.asr_engine.settings.asr_max_duration_s", 90)


# ---------------------------------------------------------------------------
# PCM helper unit tests (pure functions — no mocks needed)
# ---------------------------------------------------------------------------

class TestPcmHelpers:
    def test_duration_one_second(self):
        """1 second of 16kHz mono PCM = 16000 samples × 2 bytes = 32000 bytes."""
        from kiosk_agent.asr_engine import _pcm_bytes_to_duration, _SAMPLE_RATE
        pcm = b"\x00\x00" * _SAMPLE_RATE  # 16000 silent samples
        assert abs(_pcm_bytes_to_duration(pcm) - 1.0) < 1e-6

    def test_duration_90_seconds(self):
        from kiosk_agent.asr_engine import _pcm_bytes_to_duration, _SAMPLE_RATE
        pcm = b"\x00\x00" * (_SAMPLE_RATE * 90)
        assert abs(_pcm_bytes_to_duration(pcm) - 90.0) < 1e-6

    def test_duration_empty_buffer(self):
        from kiosk_agent.asr_engine import _pcm_bytes_to_duration
        assert _pcm_bytes_to_duration(b"") == 0.0

    def test_float32_range_max_positive(self):
        """int16 max (32767) should convert to ≈ +1.0."""
        from kiosk_agent.asr_engine import _pcm_bytes_to_float32
        pcm = struct.pack("<h", 32767)
        samples = _pcm_bytes_to_float32(pcm)
        assert len(samples) == 1
        assert 0.999 <= samples[0] <= 1.001

    def test_float32_range_max_negative(self):
        """int16 min (-32768) should convert to ≈ -1.0."""
        from kiosk_agent.asr_engine import _pcm_bytes_to_float32
        pcm = struct.pack("<h", -32768)
        samples = _pcm_bytes_to_float32(pcm)
        assert len(samples) == 1
        assert -1.001 <= samples[0] <= -0.999

    def test_float32_silent(self):
        """Silence (all zero bytes) should produce samples of 0.0."""
        from kiosk_agent.asr_engine import _pcm_bytes_to_float32
        pcm = b"\x00\x00" * 100
        samples = _pcm_bytes_to_float32(pcm)
        assert all(s == 0.0 for s in samples)

    def test_float32_odd_length_bytes(self):
        """Odd-length buffer: the trailing byte is discarded (not crashing)."""
        from kiosk_agent.asr_engine import _pcm_bytes_to_float32
        pcm = struct.pack("<h", 1000) + b"\xFF"  # 3 bytes — last byte ignored
        samples = _pcm_bytes_to_float32(pcm)
        assert len(samples) == 1  # only one complete sample


# ---------------------------------------------------------------------------
# Stub mode tests
# ---------------------------------------------------------------------------

class TestStubMode:
    def test_load_model_noop_in_stub_mode(self, stub_settings):
        """load_model() must not raise or set _model when stub mode is on."""
        from kiosk_agent import asr_engine as eng
        eng.load_model()
        assert eng._model is None

    def test_is_ready_in_stub_mode(self, stub_settings):
        from kiosk_agent.asr_engine import is_ready
        assert is_ready() is True

    @pytest.mark.parametrize("lang", ["hi", "mr", "ta"])
    def test_stub_returns_valid_result_for_all_languages(self, stub_settings, lang):
        from kiosk_agent.asr_engine import transcribe, AsrResult
        result = transcribe(b"\x00\x00" * 16000, language_hint=lang)
        assert isinstance(result, AsrResult)
        assert result.language_detected == lang
        assert len(result.transcript) > 10          # non-trivial transcript
        assert 0.0 <= result.asr_confidence <= 1.0
        assert result.asr_confidence >= 0.55        # stub always passes confidence gate

    def test_stub_default_language_hi(self, stub_settings):
        """When no language_hint is given, stub defaults to Hindi."""
        from kiosk_agent.asr_engine import transcribe
        result = transcribe(b"\x00\x00" * 16000, language_hint=None)
        assert result.language_detected == "hi"

    def test_stub_segment_confidences_non_empty(self, stub_settings):
        from kiosk_agent.asr_engine import transcribe
        result = transcribe(b"\x00\x00" * 16000, language_hint="mr")
        assert len(result.segment_confidences) > 0
        assert all(0.0 <= c <= 1.0 for c in result.segment_confidences)


# ---------------------------------------------------------------------------
# Input validation tests (run regardless of stub mode)
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_too_long_audio_raises_before_inference(self, real_settings, monkeypatch):
        """
        Audio exceeding max_duration_s must raise AsrInputTooLongError
        BEFORE any model inference is attempted.
        architecture.md §4.1: "Max input duration: 90 seconds —
        longer input is rejected with AsrInputTooLongError before inference begins."
        """
        from kiosk_agent import asr_engine as eng
        from kiosk_agent.asr_engine import AsrInputTooLongError

        # Install a mock model so the model-not-ready guard doesn't fire.
        mock_model = MagicMock()
        monkeypatch.setattr(eng, "_model", mock_model)
        monkeypatch.setattr(
            "kiosk_agent.asr_engine.settings.asr_max_duration_s", 5
        )

        # 6 seconds of PCM (above the 5s limit set above).
        too_long_pcm = b"\x00\x00" * (16000 * 6)

        with pytest.raises(AsrInputTooLongError) as exc_info:
            eng.transcribe(too_long_pcm)

        assert exc_info.value.duration_s > 5.0
        assert exc_info.value.max_s == 5
        # Model.transcribe must NOT have been called.
        mock_model.transcribe.assert_not_called()

    def test_model_not_ready_raises(self, real_settings):
        """transcribe() without a loaded model raises AsrModelNotReadyError."""
        from kiosk_agent import asr_engine as eng
        from kiosk_agent.asr_engine import AsrModelNotReadyError

        assert eng._model is None  # guaranteed by autouse fixture
        with pytest.raises(AsrModelNotReadyError):
            eng.transcribe(b"\x00\x00" * 16000)


# ---------------------------------------------------------------------------
# Mocked inference path tests
# ---------------------------------------------------------------------------

class TestMockedInference:
    """
    Tests that exercise the real transcribe() code path with a mocked
    pywhispercpp Model. The mock replicates the segment/token interface
    that the production code calls.
    """

    def _make_mock_segment(self, text: str, avg_prob: float) -> MagicMock:
        token = MagicMock()
        token.p_log = avg_prob
        seg = MagicMock()
        seg.text = text
        seg.tokens = [token, token]  # two tokens
        return seg

    def _install_mock_model(self, monkeypatch, segments, lang_str="hi"):
        import kiosk_agent.asr_engine as eng
        mock_model = MagicMock()
        mock_model.transcribe.return_value = segments
        mock_model.lang_str = lang_str
        monkeypatch.setattr(eng, "_model", mock_model)
        return mock_model

    def test_normal_transcription_returns_asrresult(
        self, real_settings, monkeypatch
    ):
        from kiosk_agent.asr_engine import transcribe, AsrResult

        segments = [
            self._make_mock_segment("सड़क पर", 0.85),
            self._make_mock_segment("बड़ा गड्ढा है।", 0.80),
        ]
        self._install_mock_model(monkeypatch, segments, lang_str="hi")

        # 3 seconds of PCM — well within limit.
        pcm = b"\x00\x00" * (16000 * 3)
        result = transcribe(pcm, language_hint="hi")

        assert isinstance(result, AsrResult)
        assert "सड़क" in result.transcript
        assert result.language_detected == "hi"
        assert result.asr_confidence > 0.55
        assert len(result.segment_confidences) == 2

    def test_low_confidence_raises_asrlowconfienceerror(
        self, real_settings, monkeypatch
    ):
        """
        When mean segment confidence < threshold (0.55), must raise
        AsrLowConfidenceError after inference completes.
        architecture.md §4.1: "if asr_confidence < 0.55, the Kiosk Agent
        routes back to the Local UI for a re-record prompt."
        """
        from kiosk_agent.asr_engine import transcribe, AsrLowConfidenceError

        segments = [self._make_mock_segment("unclear audio", 0.30)]
        self._install_mock_model(monkeypatch, segments)

        pcm = b"\x00\x00" * (16000 * 3)
        with pytest.raises(AsrLowConfidenceError) as exc_info:
            transcribe(pcm, language_hint="hi")

        assert exc_info.value.confidence < 0.55
        assert exc_info.value.threshold == 0.55

    def test_language_hint_passed_to_model(
        self, real_settings, monkeypatch
    ):
        """The language_hint must be forwarded as the `language` kwarg to Model.transcribe()."""
        from kiosk_agent.asr_engine import transcribe

        segments = [self._make_mock_segment("hello", 0.9)]
        mock_model = self._install_mock_model(monkeypatch, segments, lang_str="mr")

        pcm = b"\x00\x00" * (16000 * 2)
        transcribe(pcm, language_hint="mr")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs.get("language") == "mr"

    def test_no_language_hint_omits_language_kwarg(
        self, real_settings, monkeypatch
    ):
        """When language_hint is None, `language` kwarg must NOT be passed to Model.transcribe()."""
        from kiosk_agent.asr_engine import transcribe

        segments = [self._make_mock_segment("hello", 0.9)]
        mock_model = self._install_mock_model(monkeypatch, segments)

        pcm = b"\x00\x00" * (16000 * 2)
        transcribe(pcm, language_hint=None)

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert "language" not in call_kwargs

    def test_empty_segments_produces_empty_transcript(
        self, real_settings, monkeypatch
    ):
        """Whisper returning no segments must produce an empty transcript, not crash."""
        from kiosk_agent.asr_engine import transcribe, AsrLowConfidenceError

        self._install_mock_model(monkeypatch, [])  # no segments

        pcm = b"\x00\x00" * (16000 * 2)
        # asr_confidence will be 0.0 (below threshold) → raises low confidence
        with pytest.raises(AsrLowConfidenceError):
            transcribe(pcm)

    def test_inference_ms_recorded(
        self, real_settings, monkeypatch
    ):
        """AsrResult.inference_ms must be a non-negative integer."""
        from kiosk_agent.asr_engine import transcribe

        segments = [self._make_mock_segment("test", 0.9)]
        self._install_mock_model(monkeypatch, segments)

        pcm = b"\x00\x00" * (16000 * 2)
        result = transcribe(pcm, language_hint="hi")

        assert isinstance(result.inference_ms, int)
        assert result.inference_ms >= 0


# ---------------------------------------------------------------------------
# load_model / unload_model lifecycle
# ---------------------------------------------------------------------------

class TestModelLifecycle:
    def test_load_model_raises_if_binary_missing(
        self, real_settings, monkeypatch, tmp_path
    ):
        """load_model() must raise FileNotFoundError when the model binary is absent."""
        import kiosk_agent.asr_engine as eng

        # Point to a non-existent file.
        monkeypatch.setattr(
            "kiosk_agent.asr_engine.settings.asr_model_path",
            tmp_path / "nonexistent.bin",
        )

        with pytest.raises(FileNotFoundError):
            eng.load_model()

        assert eng._model is None  # model must not be set on failure

    def test_load_model_noop_if_already_loaded(
        self, real_settings, monkeypatch
    ):
        """Calling load_model() twice must not re-load (idempotent)."""
        import kiosk_agent.asr_engine as eng

        sentinel = MagicMock()
        monkeypatch.setattr(eng, "_model", sentinel)

        eng.load_model()
        assert eng._model is sentinel

    def test_unload_model_clears_state(self, real_settings, monkeypatch):
        import kiosk_agent.asr_engine as eng
        monkeypatch.setattr(eng, "_model", MagicMock())
        monkeypatch.setattr(eng, "_model_path", "/some/path.bin")

        eng.unload_model()
        assert eng._model is None
        assert eng._model_path == ""

    def test_is_ready_false_before_load(self, real_settings):
        from kiosk_agent.asr_engine import is_ready
        assert is_ready() is False

    def test_is_ready_true_after_mock_load(self, real_settings, monkeypatch):
        import kiosk_agent.asr_engine as eng
        monkeypatch.setattr(eng, "_model", MagicMock())
        assert eng.is_ready() is True
