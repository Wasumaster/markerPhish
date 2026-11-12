"""
MarkerPhish — Evaluation Engine
================================
Manages interaction with the local Ollama API:
    - Discover installed models
    - Build the binary (malicious vs. valid) classification prompt
    - Send it with a JSON Schema output constraint
    - Parse responses and map them to the channel-level label
    - Track per-inference latency, cold-start time, and resource metrics
    - Automatic CPU fallback on Pascal GPU CUDA errors
"""

import gc
import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from config import (
    ACTIVE_PROMPT_VARIANT,
    DEFAULT_NUM_GPU,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_NUM_CTX,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_SHOW_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    VALID_LABELS,
    binary_to_channel_label,
    build_classification_prompt,
    channel_for_source,
    get_prompt_variant,
    is_binary_match,
    to_binary_label,
)
from .utils import logger


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class InferenceResult:
    """Result of a single classification inference."""
    record_id: int
    true_label: str
    predicted_label: str
    raw_response: str
    latency_seconds: float
    prompt_eval_count: int = 0
    eval_count: int = 0
    is_binary_correct: bool = False
    true_binary: str = ""
    predicted_binary: str = ""
    channel: str = ""


@dataclass
class EvaluationRun:
    """Aggregated results of a full evaluation run."""
    model_name: str
    model_size_bytes: int = 0
    total_records: int = 0
    completed: int = 0
    failed: int = 0
    results: list[InferenceResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    prompt_variant: str = ACTIVE_PROMPT_VARIANT
    cold_start_s: float = 0.0

    @property
    def total_latency(self) -> float:
        return sum(r.latency_seconds for r in self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return self.total_latency / len(self.results)

    @property
    def elapsed_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results list to a DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "id": r.record_id,
                    "channel": r.channel,
                    "true_channel": r.true_label,
                    "predicted_channel": r.predicted_label,
                    "true_binary": r.true_binary,
                    "predicted_binary": r.predicted_binary,
                    "is_binary_correct": r.is_binary_correct,
                    "is_channel_correct": r.true_label == r.predicted_label,
                    "latency_s": round(r.latency_seconds, 4),
                    "raw_response": r.raw_response,
                }
                for r in self.results
            ]
        )


# ──────────────────────────────────────────────
# Ollama client helpers
# ──────────────────────────────────────────────
def check_ollama_connection() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[dict]:
    """
    Query the Ollama API for locally installed models.
    Returns a list of dicts with keys: name, size, modified_at, digest.
    """
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        return [
            {
                "name": m.get("name", "unknown"),
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at", ""),
                "digest": m.get("digest", "")[:12],
            }
            for m in models
        ]
    except Exception as e:
        logger.error("Failed to list Ollama models: %s", e)
        return []


def get_model_info(model_name: str) -> dict:
    """Fetch detailed model information from Ollama."""
    try:
        resp = requests.post(
            OLLAMA_SHOW_ENDPOINT,
            json={"model": model_name, "name": model_name},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Could not fetch model info for %s: %s", model_name, e)
        return {}


# Per-model cache of the "thinking" capability reported by /api/show.
_THINKING_SUPPORT_CACHE: dict[str, bool] = {}


def model_supports_thinking(model_name: str) -> bool:
    """
    Return True if Ollama reports the 'thinking' capability for the model.
    Only such models receive "think": false (other models reject the option).
    The answer is cached per model; a failed lookup is not cached.
    """
    if model_name in _THINKING_SUPPORT_CACHE:
        return _THINKING_SUPPORT_CACHE[model_name]
    info = get_model_info(model_name)
    if not info:
        return False
    supported = "thinking" in (info.get("capabilities") or [])
    _THINKING_SUPPORT_CACHE[model_name] = supported
    return supported


def build_output_schema(prompt_variant: str | None = None) -> dict:
    """
    JSON Schema passed as Ollama's "format" parameter (constrained decoding).
    The reasoning variant puts "reasoning" before "label" so the model writes
    its justification first.
    """
    settings = get_prompt_variant(prompt_variant)
    properties: dict = {}
    required: list[str] = []
    if settings["reasoning"]:
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    properties["label"] = {"type": "string", "enum": ["malicious", "valid"]}
    required.append("label")
    return {"type": "object", "properties": properties, "required": required}


def _build_options(num_gpu: int | None, num_predict: int) -> dict:
    """Ollama generation options shared by warm-up and evaluation requests."""
    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": num_predict,
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    return options


def _post_generate(payload: dict, num_gpu: int | None) -> requests.Response:
    """POST to /api/generate; on a GPU failure retry once on CPU (num_gpu=0)."""
    resp = requests.post(
        OLLAMA_GENERATE_ENDPOINT,
        json=payload,
        timeout=OLLAMA_REQUEST_TIMEOUT,
    )
    if resp.status_code != 200 and num_gpu != 0:
        logger.warning("Inference failed with status %d on GPU. Retrying with CPU (num_gpu=0)...", resp.status_code)
        payload["options"]["num_gpu"] = 0
        resp = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
    return resp


def warm_up_model(model_name: str, num_gpu: int | None = DEFAULT_NUM_GPU) -> float:
    """
    Load the model into memory with one tiny request before the timed run.

    Uses the same num_ctx and num_gpu as the evaluation (otherwise Ollama would
    reload the model on the first real request). Returns Ollama's load_duration
    in seconds (wall time if it is not reported, 0.0 on error).
    """
    payload = {
        "model": model_name,
        "prompt": "Hi",
        "stream": False,
        "options": _build_options(num_gpu, num_predict=1),
    }
    if model_supports_thinking(model_name):
        payload["think"] = False

    start = time.perf_counter()
    try:
        resp = _post_generate(payload, num_gpu)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Model warm-up failed for %s: %s", model_name, e)
        return 0.0
    wall = time.perf_counter() - start

    load_ns = data.get("load_duration")
    if isinstance(load_ns, (int, float)) and load_ns > 0:
        return load_ns / 1e9
    return wall


# ──────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────
def parse_label(raw_response: str) -> str:
    """
    Extract a channel-level label (phishing / smishing / valid) from free text.

    Kept for reports and callers from the three-label era; parse_binary_label
    uses it as its last free-text fallback.

    Strategy:
        1. Exact match after lowering + stripping markdown/punctuation
        2. Search for known labels within the response text
        3. Map common synonyms (legitimate -> valid, safe -> valid, etc.)
        4. Fallback to 'unknown'
    """
    cleaned = raw_response.strip().lower()
    cleaned = re.sub(r'[*_`"\'\s:.-]+', " ", cleaned).strip()

    # Exact token match
    tokens = cleaned.split()
    if tokens:
        first_token = tokens[0]
        if first_token in VALID_LABELS:
            return first_token

    # Check for direct presence of labels
    for label in ["phishing", "smishing", "valid"]:
        if label in cleaned:
            return label

    # Check for common synonyms
    synonym_map = {
        "legitimate": "valid",
        "safe": "valid",
        "ham": "valid",
        "benign": "valid",
        "not malicious": "valid",
        "normal": "valid",
        "clean": "valid",
        "phish": "phishing",
        "sms phishing": "smishing",
        "smish": "smishing",
    }
    for synonym, label in synonym_map.items():
        if synonym in cleaned:
            return label

    logger.debug("Could not parse label from response: '%s'", raw_response[:100])
    return "unknown"


_MALICIOUS_WORDS = {"malicious", "phishing", "smishing", "scam", "fraud", "fraudulent", "spam"}
_VALID_WORDS = {"valid", "safe", "legitimate", "benign", "ham"}
_NEGATED_THREAT_RE = re.compile(
    r"\b(?:not|isn't|isnt|is\s+not|no)\s+(?:a\s+|an\s+)?(?:malicious|phishing|smishing|scam|fraud|fraudulent|spam)\b"
)
_LABEL_FIELD_RE = re.compile(r"[\"']?label[\"']?\s*[:=]\s*[\"']?\s*([A-Za-z][A-Za-z ]*)", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalize_binary_token(value) -> str:
    """Map a single label value to 'malicious', 'valid', or 'unknown'."""
    token = re.sub(r"[^a-z ]+", " ", str(value).lower()).strip()
    token = re.sub(r"\s+", " ", token)
    if _NEGATED_THREAT_RE.search(token):
        return "valid"
    if token in _MALICIOUS_WORDS:
        return "malicious"
    if token in _VALID_WORDS:
        return "valid"
    return "unknown"


def parse_binary_label(raw_response: str) -> str:
    """
    Extract the binary decision ('malicious' / 'valid') from a model response.

    Strategy:
        1. Parse JSON (also inside ``` code fences) and read its "label" field
        2. Regex on a "label": value field in malformed JSON
        3. Free text: "not malicious" -> valid, "malicious" -> malicious,
           then the legacy parse_label() mapped to binary, then synonyms
        4. Fallback to 'unknown'
    """
    if not isinstance(raw_response, str) or not raw_response.strip():
        return "unknown"
    text = raw_response.strip()

    # 1. JSON
    candidates = [m.strip() for m in _CODE_FENCE_RE.findall(text)]
    candidates.append(text)
    obj_match = _JSON_OBJECT_RE.search(text)
    if obj_match:
        candidates.append(obj_match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).strip().lower() == "label":
                    mapped = _normalize_binary_token(value)
                    if mapped != "unknown":
                        return mapped

    # 2. "label" field in malformed JSON
    field_match = _LABEL_FIELD_RE.search(text)
    if field_match:
        mapped = _normalize_binary_token(field_match.group(1))
        if mapped != "unknown":
            return mapped

    # 3. Free text
    lowered = text.lower()
    if _NEGATED_THREAT_RE.search(lowered):
        return "valid"
    if re.search(r"\bmalicious\b", lowered):
        return "malicious"
    legacy = parse_label(text)
    if legacy in ("phishing", "smishing"):
        return "malicious"
    if legacy == "valid":
        return "valid"
    words = set(re.findall(r"[a-z]+", lowered))
    if words & _MALICIOUS_WORDS:
        return "malicious"
    if words & _VALID_WORDS:
        return "valid"

    logger.debug("Could not parse binary label from response: '%s'", raw_response[:100])
    return "unknown"


# ──────────────────────────────────────────────
# Inference engine
# ──────────────────────────────────────────────
def classify_message(
    model_name: str,
    message_text: str,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    channel: str | None = None,
    prompt_variant: str | None = None,
) -> tuple[str, str, float, dict]:
    """
    Send a single message to Ollama for binary classification (malicious vs. valid).
    Supports setting num_gpu (0 = CPU mode to prevent CUDA toolchain buffer overrun).

    The channel ('email' / 'sms', known from the dataset source) is stated in the
    prompt and used to map the binary answer to the channel-level label.

    Returns
    -------
    tuple of (predicted_label, raw_response, latency_seconds, metadata)
    where predicted_label is 'phishing', 'smishing', 'valid', or 'unknown'.
    """
    variant = prompt_variant or ACTIVE_PROMPT_VARIANT
    settings = get_prompt_variant(variant)
    prompt = build_classification_prompt(message_text, channel, variant)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": build_output_schema(variant),
        "options": _build_options(num_gpu, settings["num_predict"]),
    }
    if model_supports_thinking(model_name):
        payload["think"] = False

    start = time.perf_counter()
    try:
        resp = _post_generate(payload, num_gpu)
        resp.raise_for_status()
        data = resp.json()
    except requests.Timeout:
        elapsed = time.perf_counter() - start
        logger.warning("Timeout after %.1fs for model %s", elapsed, model_name)
        return "unknown", "[TIMEOUT]", elapsed, {}
    except requests.ConnectionError:
        elapsed = time.perf_counter() - start
        logger.error("Cannot connect to Ollama — is it running?")
        return "unknown", "[CONNECTION_ERROR: Ollama offline]", elapsed, {}
    except Exception as e:
        elapsed = time.perf_counter() - start
        error_msg = str(e)
        # Capture error response body if available
        if hasattr(e, "response") and e.response is not None:
            try:
                error_msg = f"{e.response.status_code} — {e.response.json().get('error', e.response.text[:200])}"
            except Exception:
                error_msg = f"{e.response.status_code} — {e.response.text[:200]}"
        logger.error("Inference error: %s", error_msg)
        return "unknown", f"[ERROR: {error_msg}]", elapsed, {}

    latency = time.perf_counter() - start
    raw_response = data.get("response", "").strip()
    binary = parse_binary_label(raw_response)
    predicted = binary_to_channel_label(binary, channel)

    metadata = {
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": data.get("eval_count", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "load_duration_ns": data.get("load_duration", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "binary_label": binary,
        "channel": channel or "unknown",
        "prompt_variant": variant,
    }

    return predicted, raw_response, latency, metadata


def run_evaluation(
    model_name: str,
    df: pd.DataFrame,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    progress_callback=None,
    prompt_variant: str | None = None,
) -> EvaluationRun:
    """
    Run a full evaluation of a model against the provided dataset.

    The model is warmed up first, so the timed run (and the first latency)
    excludes model loading; the load time is stored as cold_start_s.
    """
    variant = prompt_variant or ACTIVE_PROMPT_VARIANT
    get_prompt_variant(variant)  # validate early
    run = EvaluationRun(model_name=model_name, total_records=len(df), prompt_variant=variant)

    # Fetch model size
    models = list_available_models()
    for m in models:
        if m["name"] == model_name or m["name"].startswith(model_name):
            run.model_size_bytes = m["size"]
            break

    logger.info(
        "Starting evaluation: model=%s, records=%d, num_gpu=%s, prompt_variant=%s",
        model_name,
        len(df),
        num_gpu,
        variant,
    )
    run.cold_start_s = warm_up_model(model_name, num_gpu)
    logger.info("Model warm-up finished (cold start %.2fs)", run.cold_start_s)
    run.start_time = time.time()

    has_source = "source" in df.columns
    for idx, row in df.iterrows():
        record_id = row["id"]
        text = row["text"]
        true_label = row["true_label"]
        channel = channel_for_source(row["source"] if has_source else None, true_label)

        predicted, raw_resp, latency, meta = classify_message(
            model_name=model_name,
            message_text=text,
            num_gpu=num_gpu,
            channel=channel,
            prompt_variant=variant,
        )

        is_bin_correct = is_binary_match(true_label, predicted)
        result = InferenceResult(
            record_id=record_id,
            true_label=true_label,
            predicted_label=predicted,
            raw_response=raw_resp,
            latency_seconds=latency,
            prompt_eval_count=meta.get("prompt_eval_count", 0),
            eval_count=meta.get("eval_count", 0),
            is_binary_correct=is_bin_correct,
            true_binary=to_binary_label(true_label),
            predicted_binary=to_binary_label(predicted),
            channel=channel,
        )

        run.results.append(result)
        run.completed += 1
        if predicted == "unknown":
            run.failed += 1

        if progress_callback:
            progress_callback(run.completed, run.total_records, result)

        # Aggressive memory cleanup every 25 messages
        if run.completed % 25 == 0:
            gc.collect()

    run.end_time = time.time()
    logger.info(
        "Evaluation complete: %d/%d processed (%.1fs, avg %.2fs/msg, failed=%d)",
        run.completed,
        run.total_records,
        run.elapsed_time,
        run.avg_latency,
        run.failed,
    )
    return run
