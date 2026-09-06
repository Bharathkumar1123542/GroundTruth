"""
kiosk_agent/structuring_engine.py
-----------------------------------
Structuring Engine — converts ASR transcript → schema-conformant StructuredComplaint.

Specification: architecture.md §4.2, implementation.md §4.3
  Model    : Gemma-2-2b-it, LoRA fine-tuned (rank 16, q_proj/v_proj),
             quantized to Q4_K_M GGUF (~1.6 GB), served via llama.cpp.
  Grammar  : structured_complaint.gbnf loaded at process start;
             GBNF-constrained decoding makes invalid output structurally
             unreachable (ADR-002).
  Input    : ASR transcript (str) + detected language code (str).
  Output   : StructuredComplaint (architecture.md §6.1).
  Timeout  : StructuringTimeoutError if generation exceeds 20 seconds.
  Stub mode: returns a deterministic StructuredComplaint when
             STUB_STRUCTURING=true (no GGUF binary required).

Prompt templates:
  Per-language prompt files live in kiosk_agent/prompts/{lang}.txt.
  The grammar file is shared across languages — only the prompt changes.
  Adding a new language requires only a new prompts/{lang}.txt file and a
  new Whisper language code entry — no code change (architecture.md §13 NFR).

Model loading:
  load_model() must be called once at Kiosk Agent startup (main.py lifespan).
  Model and grammar are loaded once and held in module-level state.
  Per implementation.md §4.3: "StructuringTimeoutError if generation exceeds 20 seconds."
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from kiosk_agent.config import settings
from kiosk_agent.schemas import (
    Category,
    StructuredComplaint,
    CATEGORY_TO_DEPARTMENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class StructuringError(Exception):
    """Base class for all Structuring Engine errors."""

class StructuringTimeoutError(StructuringError):
    """
    Raised when LLM generation exceeds settings.structuring_timeout_s (default 20s).
    architecture.md §9 latency budget: structuring ≈ 15s for 200-token output.
    """
    def __init__(self, timeout_s: int) -> None:
        super().__init__(f"Structuring Engine timed out after {timeout_s}s.")
        self.timeout_s = timeout_s

class StructuringModelNotReadyError(StructuringError):
    """Raised if structure() is called before load_model()."""

class StructuringOutputInvalidError(StructuringError):
    """
    Raised if the model output, despite GBNF constraints, fails Pydantic
    validation. Should be extremely rare — indicates a grammar/schema mismatch.
    """


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_llm = None          # llama_cpp.Llama instance
_grammar = None      # llama_cpp.LlamaGrammar instance
_grammar_str: str = ""


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------

_PROMPT_CACHE: dict[str, str] = {}

# Default fallback prompt template (English) used when a language-specific
# template is not found. The {transcript} placeholder is replaced at call time.
_DEFAULT_PROMPT_TEMPLATE = """\
You are a civic complaint structuring assistant.
Convert the following complaint transcript into a structured JSON record.
Output ONLY valid JSON conforming to the schema — no prose, no explanation.

Transcript: {transcript}

JSON:"""


def _load_prompt_template(language: str) -> str:
    """
    Load the prompt template for a given language from prompts/{language}.txt.
    Results are cached after first load. Falls back to the default English
    template if the file does not exist (degraded mode, not a hard failure).
    """
    if language in _PROMPT_CACHE:
        return _PROMPT_CACHE[language]

    prompts_dir = settings.prompts_dir.resolve()
    lang_file = prompts_dir / f"{language}.txt"

    if lang_file.exists():
        template = lang_file.read_text(encoding="utf-8").strip()
        logger.info("Loaded prompt template for language=%s from %s", language, lang_file)
    else:
        logger.warning(
            "Prompt template not found for language=%s at %s — using default.",
            language, lang_file,
        )
        template = _DEFAULT_PROMPT_TEMPLATE

    _PROMPT_CACHE[language] = template
    return template


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the GGUF model and GBNF grammar into memory.

    Must be called once at Kiosk Agent startup (main.py lifespan).
    Subsequent calls are no-ops if the model is already loaded.
    In stub mode (settings.stub_structuring == True), this is a no-op.
    """
    global _llm, _grammar, _grammar_str

    if _llm is not None:
        logger.debug("Structuring Engine: model already loaded — skipping.")
        return

    if settings.stub_structuring:
        logger.info("Structuring Engine: stub mode active — model not loaded.")
        return

    # --- Load GBNF grammar ---
    grammar_path = settings.grammar_path.resolve()
    if not grammar_path.exists():
        raise FileNotFoundError(
            f"GBNF grammar file not found at {grammar_path}. "
            "Ensure kiosk_agent/grammar/structured_complaint.gbnf is present."
        )
    _grammar_str = grammar_path.read_text(encoding="utf-8")

    # --- Load GGUF model ---
    model_path = str(settings.llm_model_path.resolve())
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Structuring Engine model binary not found at {model_path}."
        )
    logger.info("Loading Structuring Engine model from %s …", model_path)
    t0 = time.perf_counter()

    try:
        # llama_cpp is imported lazily so the module can be imported
        # in test environments without llama-cpp-python installed.
        from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

        _grammar = LlamaGrammar.from_string(_grammar_str)

        _llm = Llama(
            model_path=model_path,
            n_ctx=1024,           # context window (transcript + output)
            n_threads=4,          # CPU threads
            n_gpu_layers=0,       # 0 = CPU only; set >0 for GPU offload on Jetson
            verbose=False,
            use_mmap=True,        # memory-map the model file (reduces RAM copy)
            use_mlock=False,      # don't lock pages (conserve memory on Pi-class HW)
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Structuring Engine model loaded in %d ms.", elapsed_ms)

    except FileNotFoundError:
        logger.error(
            "LLM model binary not found at %s. "
            "Run scripts/fetch_models.sh or set LLM_MODEL_PATH.",
            model_path,
        )
        raise
    except ImportError:
        logger.error(
            "llama-cpp-python is not installed. Run `poetry install` inside kiosk/."
        )
        raise


def unload_model() -> None:
    """Release model from memory. Used in tests to reset global state."""
    global _llm, _grammar, _grammar_str
    _llm = None
    _grammar = None
    _grammar_str = ""


def is_ready() -> bool:
    """Return True if model is loaded, or stub mode is active."""
    return settings.stub_structuring or _llm is not None


# ---------------------------------------------------------------------------
# Core structuring function
# ---------------------------------------------------------------------------

def structure(transcript: str, language: str) -> StructuredComplaint:
    """
    Convert a transcript string into a schema-conformant StructuredComplaint.

    Parameters
    ----------
    transcript : str
        The raw ASR transcript text.
    language : str
        ISO-639-1 language code of the transcript ("hi", "mr", "ta").

    Returns
    -------
    StructuredComplaint
        Schema-conformant structured complaint (architecture.md §6.1).

    Raises
    ------
    StructuringModelNotReadyError
        If load_model() has not been called and stub mode is off.
    StructuringTimeoutError
        If generation exceeds settings.structuring_timeout_s (default 20s).
    StructuringOutputInvalidError
        If the grammar-constrained output fails Pydantic validation
        (grammar/schema mismatch — should be extremely rare).
    """
    # ── Stub mode ─────────────────────────────────────────────────────────
    if settings.stub_structuring:
        return _stub_structure(transcript, language)

    # ── Guard ─────────────────────────────────────────────────────────────
    if _llm is None:
        raise StructuringModelNotReadyError(
            "Structuring Engine model is not loaded. Call load_model() at startup."
        )

    # ── Build prompt ──────────────────────────────────────────────────────
    template = _load_prompt_template(language)
    prompt = template.format(transcript=transcript)

    logger.info(
        "Structuring: starting generation  language=%s  transcript_len=%d",
        language, len(transcript),
    )

    # ── Run constrained generation with timeout ───────────────────────────
    timeout_s = settings.structuring_timeout_s
    result_container: list[dict | None] = [None]
    exception_container: list[BaseException | None] = [None]

    def _generate() -> None:
        try:
            output = _llm(  # type: ignore[misc]
                prompt,
                grammar=_grammar,
                max_tokens=512,      # sufficient for the structured complaint schema
                temperature=0.0,     # deterministic / greedy (ADR-002: no randomness)
                top_p=1.0,
                repeat_penalty=1.0,
                stop=["}\n", "} "],  # stop after the closing brace
            )
            result_container[0] = output
        except Exception as exc:  # noqa: BLE001
            exception_container[0] = exc

    t0 = time.perf_counter()
    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if thread.is_alive():
        # Thread is still running — generation timed out.
        # We cannot forcibly kill the thread, but marking it daemon ensures
        # it won't block process exit. The Kiosk Agent will surface a timeout
        # error to the resident for a re-record prompt.
        logger.error(
            "Structuring Engine timed out after %ds (elapsed=%dms).",
            timeout_s, elapsed_ms,
        )
        raise StructuringTimeoutError(timeout_s)

    if exception_container[0] is not None:
        raise StructuringError(
            f"LLM generation failed: {exception_container[0]}"
        ) from exception_container[0]

    raw_output = result_container[0]
    if raw_output is None:
        raise StructuringError("LLM returned no output.")

    # ── Parse + validate output ───────────────────────────────────────────
    # llama_cpp returns a dict with choices[0].text containing the generated text.
    generated_text: str = raw_output["choices"][0]["text"].strip()

    # Ensure the JSON object is complete (sometimes stop tokens leave a partial).
    if not generated_text.endswith("}"):
        generated_text = generated_text.rstrip() + "}"

    logger.info(
        "Structuring: generation complete  elapsed_ms=%d  output_len=%d",
        elapsed_ms, len(generated_text),
    )
    logger.debug("Structuring raw output: %s", generated_text[:200])

    try:
        data = json.loads(generated_text)
    except json.JSONDecodeError as exc:
        raise StructuringOutputInvalidError(
            f"Generated text is not valid JSON despite GBNF constraints: {exc}\n"
            f"Output: {generated_text[:300]}"
        ) from exc

    try:
        complaint = StructuredComplaint(**data)
    except Exception as exc:  # noqa: BLE001
        raise StructuringOutputInvalidError(
            f"Generated JSON does not conform to StructuredComplaint schema: {exc}\n"
            f"Data: {data}"
        ) from exc

    logger.info(
        "Structuring result: category=%s  confidence=%.3f",
        complaint.category, complaint.structuring_confidence,
    )
    return complaint


# ---------------------------------------------------------------------------
# Stub structuring
# ---------------------------------------------------------------------------

_STUB_DESCRIPTIONS: dict[str, str] = {
    "hi": "मुख्य सड़क पर सामुदायिक केंद्र के पास बड़ा गड्ढा होने से वाहनों को नुकसान।",
    "mr": "मुख्य रस्त्यावर सामुदायिक केंद्राजवळ मोठा खड्डा, वाहनांचे नुकसान होत आहे.",
    "ta": "சமுதாய மையத்திற்கு அருகில் முக்கிய சாலையில் பெரிய குழி — வாகனங்கள் சேதமடைகின்றன.",
}


def _stub_structure(transcript: str, language: str) -> StructuredComplaint:
    """
    Returns a deterministic, schema-conformant StructuredComplaint
    without invoking llama.cpp. Used when STUB_STRUCTURING=true.
    The description is language-tagged so the confirmation screen renders
    a realistic localised stub for hackathon demo purposes.
    """
    desc = _STUB_DESCRIPTIONS.get(language, _STUB_DESCRIPTIONS["hi"])
    # Truncate transcript-derived location hint to schema max (64 chars).
    location_hint = transcript[:60].strip() if transcript else "near community centre"

    logger.debug(
        "Structuring Engine: returning stub StructuredComplaint  language=%s", language
    )
    return StructuredComplaint(
        category=Category.ROAD,
        subcategory="Pothole",
        description=desc[:280],
        location_hint=location_hint,
        reported_asset_type="road_segment",
        urgency_keywords=["pothole", "damage", "road"],
        structuring_confidence=0.91,
    )
