"""
kiosk_agent/asr_engine.py
--------------------------
ASR Engine — quantized Whisper via whisper.cpp (pywhispercpp bindings).

Specification: architecture.md §4.1
  Model  : whisper-small multilingual, quantized INT8 GGML
           (ggml-small-q8_0.bin, ~500 MB)
  Input  : 16 kHz mono PCM audio bytes (max 90 seconds, enforced here)
  Output : AsrResult(transcript, language_detected, asr_confidence,
                     segment_confidences)
  Errors :
    AsrInputTooLongError   — audio exceeds max_duration_s before inference
    AsrLowConfidenceError  — asr_confidence < threshold (default 0.55)
    AsrModelNotReadyError  — model not loaded (should never reach production)

Model loading:
  `load_model()` must be called once at process start (Kiosk Agent lifespan).
  Per implementation.md §4.2: "Model load happens once at process start
  (not per-request) to keep p95 latency within the 90s end-to-end budget."

Stub mode:
  When STUB_STRUCTURING=true (which also implies no real model binary),
  `transcribe()` returns a deterministic stub AsrResult without invoking
  pywhispercpp. This allows Phase 1 UI and Phase 3 structuring tests to run
  before the model binary is available.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from kiosk_agent.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions (architecture.md §4.1, implementation.md §4.2)
# ---------------------------------------------------------------------------

class AsrError(Exception):
    """Base class for all ASR Engine errors."""

class AsrInputTooLongError(AsrError):
    """
    Raised when the audio buffer exceeds asr_max_duration_s before inference.
    The Kiosk Agent catches this and shows a re-record prompt.
    """
    def __init__(self, duration_s: float, max_s: int) -> None:
        super().__init__(
            f"Audio duration {duration_s:.1f}s exceeds maximum {max_s}s."
        )
        self.duration_s = duration_s
        self.max_s = max_s

class AsrLowConfidenceError(AsrError):
    """
    Raised when asr_confidence < settings.asr_confidence_threshold (default 0.55).
    The Kiosk Agent routes back to the Local UI for a re-record prompt
    (architecture.md §4.1).
    """
    def __init__(self, confidence: float, threshold: float) -> None:
        super().__init__(
            f"ASR confidence {confidence:.3f} below threshold {threshold:.3f}."
        )
        self.confidence = confidence
        self.threshold = threshold

class AsrModelNotReadyError(AsrError):
    """Raised if transcribe() is called before load_model()."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AsrResult:
    """
    Output of the ASR Engine (architecture.md §4.1).
    Maps directly to the JSON schema:
      { transcript, language_detected, segment_confidences, asr_confidence }
    """
    transcript:          str
    language_detected:   str                     # ISO-639-1 e.g. "hi", "mr", "ta"
    asr_confidence:      float                   # 0.0–1.0 aggregate confidence
    segment_confidences: list[float] = field(default_factory=list)
    duration_s:          float = 0.0             # audio duration (informational)
    inference_ms:        int   = 0               # wall-clock inference time (ms)


# ---------------------------------------------------------------------------
# Module-level model state
# ---------------------------------------------------------------------------

_model = None          # pywhispercpp.Model instance (None until load_model())
_model_path: str = ""  # path the model was loaded from


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the quantized Whisper model into memory.

    Must be called once at Kiosk Agent startup (main.py lifespan).
    Subsequent calls are no-ops if the model is already loaded.

    In stub mode (settings.stub_structuring == True), this is also a no-op —
    the model binary may not be present.
    """
    global _model, _model_path

    if _model is not None:
        logger.debug("ASR model already loaded — skipping.")
        return

    if settings.stub_structuring:
        logger.info("ASR Engine: stub mode active — model not loaded.")
        return

    model_path = str(settings.asr_model_path.resolve())
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"ASR model binary not found at {model_path}. "
            "Run scripts/fetch_models.sh or set ASR_MODEL_PATH."
        )
    logger.info("Loading ASR model from %s …", model_path)
    t0 = time.perf_counter()

    try:
        from pywhispercpp.model import Model  # noqa: PLC0415

        _model = Model(
            model_path,
            n_threads=4,         # use 4 CPU threads (safe on Jetson Orin Nano)
            print_progress=False,
            print_realtime=False,
        )
        _model_path = model_path
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("ASR model loaded in %d ms.", elapsed_ms)

    except FileNotFoundError:
        logger.error(
            "ASR model binary not found at %s. "
            "Run scripts/fetch_models.sh or set ASR_MODEL_PATH.",
            model_path,
        )
        raise
    except ImportError:
        logger.error(
            "pywhispercpp is not installed. "
            "Run `poetry install` inside kiosk/."
        )
        raise


def unload_model() -> None:
    """Release the model from memory (used in tests to reset global state)."""
    global _model, _model_path
    _model = None
    _model_path = ""


def is_ready() -> bool:
    """Return True if the model is loaded and ready, or if stub mode is active."""
    return settings.stub_structuring or _model is not None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_PCM16_BYTES_PER_SAMPLE = 2    # int16 = 2 bytes
_SAMPLE_RATE             = 16000  # Hz — Whisper's expected input rate

def _pcm_bytes_to_duration(audio_bytes: bytes) -> float:
    """Return duration in seconds for a raw 16-bit, 16 kHz, mono PCM buffer."""
    num_samples = len(audio_bytes) // _PCM16_BYTES_PER_SAMPLE
    return num_samples / _SAMPLE_RATE


def _pcm_bytes_to_float32(audio_bytes: bytes) -> list[float]:
    """
    Convert raw 16-bit signed PCM bytes to a list of float32 samples in [-1, 1].
    whisper.cpp's Python bindings accept a list[float] (or numpy array).
    """
    num_samples = len(audio_bytes) // _PCM16_BYTES_PER_SAMPLE
    samples = struct.unpack(f"<{num_samples}h", audio_bytes[:num_samples * 2])
    return [s / 32768.0 for s in samples]


def _webm_or_ogg_to_pcm16(audio_bytes: bytes) -> bytes:
    """
    Convert WebM/OGG/Opus audio (from MediaRecorder) to 16-bit 16 kHz mono PCM.

    Uses ffmpeg subprocess for broad format support. ffmpeg is expected to be
    available on the kiosk OS (Ubuntu 22.04: `sudo apt install ffmpeg`).

    Falls back to returning the input bytes unchanged if ffmpeg is unavailable,
    which will cause an inference error rather than a silent bad result.
    """
    import subprocess  # noqa: PLC0415
    import tempfile    # noqa: PLC0415
    import os          # noqa: PLC0415

    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as src:
            src.write(audio_bytes)
            src_path = src.name

        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as dst:
            dst_path = dst.name

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", str(_SAMPLE_RATE),
                "-ac", "1",          # mono
                "-f", "s16le",       # raw 16-bit signed little-endian PCM
                dst_path,
            ],
            capture_output=True,
            timeout=15,
        )

        if result.returncode != 0:
            logger.warning(
                "ffmpeg conversion failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:200],
            )
            return audio_bytes  # fall through to inference (will likely fail)

        with open(dst_path, "rb") as f:
            pcm = f.read()

        return pcm

    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("ffmpeg unavailable or timed out (%s) — passing raw bytes.", exc)
        return audio_bytes
    finally:
        for path in (src_path, dst_path):  # type: ignore[possibly-undefined]
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Core transcription function
# ---------------------------------------------------------------------------

def transcribe(
    audio_bytes: bytes,
    language_hint: str | None = None,
) -> AsrResult:
    """
    Transcribe audio bytes to text using the quantized Whisper model.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio from the browser MediaRecorder (WebM/OGG/Opus) or
        raw 16-bit PCM (for testing). Automatically converted to PCM16.
    language_hint : str | None
        Optional ISO-639-1 code to bias Whisper's language detection.
        None means auto-detect.

    Returns
    -------
    AsrResult
        Transcript and confidence metadata.

    Raises
    ------
    AsrInputTooLongError
        If the decoded audio duration exceeds asr_max_duration_s.
    AsrLowConfidenceError
        If the computed asr_confidence is below asr_confidence_threshold.
    AsrModelNotReadyError
        If load_model() has not been called and stub mode is off.
    """
    # ── Stub mode ────────────────────────────────────────────────────────
    if settings.stub_structuring:
        return _stub_transcribe(language_hint)

    # ── Guard: model must be loaded ───────────────────────────────────────
    if _model is None:
        raise AsrModelNotReadyError(
            "ASR model is not loaded. Call load_model() at startup."
        )

    # ── Convert to PCM16 if needed ────────────────────────────────────────
    # Detect WebM/OGG magic bytes; if not raw PCM, convert via ffmpeg.
    if audio_bytes[:4] in (b"\x1a\x45\xdf\xa3", b"OggS"):
        logger.debug("Detected container format — converting to PCM16 via ffmpeg.")
        audio_bytes = _webm_or_ogg_to_pcm16(audio_bytes)

    # ── Duration check (pre-inference) ────────────────────────────────────
    duration_s = _pcm_bytes_to_duration(audio_bytes)
    max_s = settings.asr_max_duration_s
    if duration_s > max_s:
        raise AsrInputTooLongError(duration_s, max_s)

    logger.info(
        "ASR: starting transcription  duration=%.1fs  language_hint=%s",
        duration_s, language_hint,
    )
    t0 = time.perf_counter()

    # ── Run inference ─────────────────────────────────────────────────────
    float_samples = _pcm_bytes_to_float32(audio_bytes)

    # pywhispercpp.Model.transcribe() returns a list of Segment objects.
    # Each segment has .text and optionally .tokens with per-token probabilities.
    kwargs: dict = {"n_threads": 4}
    if language_hint:
        kwargs["language"] = language_hint

    segments = _model.transcribe(float_samples, **kwargs)  # type: ignore[union-attr]

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("ASR: inference complete  elapsed_ms=%d", elapsed_ms)

    # ── Assemble result ───────────────────────────────────────────────────
    full_transcript = " ".join(
        seg.text.strip() for seg in segments if seg.text.strip()
    )

    # Compute per-segment confidence from average token log-probabilities.
    # whisper.cpp exposes log-prob per token via segment.tokens.
    segment_confidences: list[float] = []
    for seg in segments:
        if hasattr(seg, "tokens") and seg.tokens:
            # avg log-prob → probability (clamped to [0, 1])
            avg_logprob = sum(t.p_log for t in seg.tokens) / len(seg.tokens)
            prob = min(1.0, max(0.0, avg_logprob))
            segment_confidences.append(round(prob, 4))
        else:
            segment_confidences.append(0.7)  # fallback neutral confidence

    asr_confidence = (
        sum(segment_confidences) / len(segment_confidences)
        if segment_confidences else 0.0
    )

    # Detect language from the model (pywhispercpp exposes model.lang_str).
    detected_lang = language_hint or "hi"  # fallback if not exposed by binding
    if hasattr(_model, "lang_str"):
        detected_lang = cast(str, _model.lang_str)

    result = AsrResult(
        transcript=full_transcript,
        language_detected=detected_lang,
        asr_confidence=round(asr_confidence, 4),
        segment_confidences=segment_confidences,
        duration_s=round(duration_s, 2),
        inference_ms=elapsed_ms,
    )

    # ── Confidence gate (architecture.md §4.1) ────────────────────────────
    threshold = settings.asr_confidence_threshold
    if asr_confidence < threshold:
        logger.warning(
            "ASR low confidence: %.3f < %.3f  transcript='%s...'",
            asr_confidence, threshold, full_transcript[:60],
        )
        raise AsrLowConfidenceError(asr_confidence, threshold)

    logger.info(
        "ASR result: confidence=%.3f  language=%s  words=%d  transcript='%s...'",
        asr_confidence,
        detected_lang,
        len(full_transcript.split()),
        full_transcript[:80],
    )
    return result


# ---------------------------------------------------------------------------
# Stub transcription (development / pre-model-binary)
# ---------------------------------------------------------------------------

_STUB_TRANSCRIPTS: dict[str, str] = {
    "hi": "मुख्य सड़क पर सामुदायिक केंद्र के पास एक बड़ा गड्ढा है जिससे वाहनों को नुकसान हो रहा है।",
    "mr": "मुख्य रस्त्यावर सामुदायिक केंद्राजवळ एक मोठा खड्डा आहे ज्यामुळे वाहनांचे नुकसान होत आहे.",
    "ta": "சமுதாய மையத்திற்கு அருகில் உள்ள முக்கிய சாலையில் ஒரு பெரிய குழி உள்ளது, இது வாகனங்களை சேதப்படுத்துகிறது.",
}

def _stub_transcribe(language_hint: str | None) -> AsrResult:
    """
    Returns a deterministic stub AsrResult without invoking pywhispercpp.
    Used when STUB_STRUCTURING=true (model binary unavailable in dev).
    """
    lang = language_hint or "hi"
    transcript = _STUB_TRANSCRIPTS.get(lang, _STUB_TRANSCRIPTS["hi"])
    logger.debug("ASR Engine: returning stub transcript for language=%s", lang)
    return AsrResult(
        transcript=transcript,
        language_detected=lang,
        asr_confidence=0.88,  # above the 0.55 threshold — stub always passes
        segment_confidences=[0.88, 0.90, 0.85],
        duration_s=12.0,
        inference_ms=50,
    )
