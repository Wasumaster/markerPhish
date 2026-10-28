"""
MarkerPhish — Evaluation Engine
================================
Manages interaction with the local Ollama API:
    - Discover installed models
    - Send classification prompts
    - Parse responses and extract labels
    - Track per-inference latency and resource metrics
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
    CLASSIFICATION_PROMPT,
    DEFAULT_NUM_GPU,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_SHOW_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    VALID_LABELS,
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
            json={"name": model_name},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Could not fetch model info for %s: %s", model_name, e)
        return {}


# ──────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────
def parse_label(raw_response: str) -> str:
    """
    Extract a classification label from the model's raw response.

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


# ──────────────────────────────────────────────
# Inference engine
# ──────────────────────────────────────────────
def classify_message(
    model_name: str,
    message_text: str,
    num_gpu: int | None = DEFAULT_NUM_GPU,
) -> tuple[str, str, float, dict]:
    """
    Send a single message to Ollama for classification.
    Supports setting num_gpu (0 = CPU mode to prevent CUDA toolchain buffer overrun).

    Returns
    -------
    tuple of (predicted_label, raw_response, latency_seconds, metadata)
    """
    prompt = CLASSIFICATION_PROMPT.format(message_text=message_text)

    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": OLLAMA_NUM_PREDICT,
    }
    if num_gpu is not None:
        options["num_gpu"] = num_gpu

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            timeout=OLLAMA_REQUEST_TIMEOUT,
        )
        # If server failed and num_gpu was not 0, attempt fallback to CPU (num_gpu=0)
        if resp.status_code != 200 and num_gpu != 0:
            logger.warning("Inference failed with status %d on GPU. Retrying with CPU (num_gpu=0)...", resp.status_code)
            payload["options"]["num_gpu"] = 0
            resp = requests.post(
                OLLAMA_GENERATE_ENDPOINT,
                json=payload,
                timeout=OLLAMA_REQUEST_TIMEOUT,
            )

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
    predicted = parse_label(raw_response)

    metadata = {
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": data.get("eval_count", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "load_duration_ns": data.get("load_duration", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
    }

    return predicted, raw_response, latency, metadata


def run_evaluation(
    model_name: str,
    df: pd.DataFrame,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    progress_callback=None,
) -> EvaluationRun:
    """
    Run a full evaluation of a model against the provided dataset.
    """
    run = EvaluationRun(model_name=model_name, total_records=len(df))

    # Fetch model size
    models = list_available_models()
    for m in models:
        if m["name"] == model_name or m["name"].startswith(model_name):
            run.model_size_bytes = m["size"]
            break

    logger.info(
        "Starting evaluation: model=%s, records=%d, num_gpu=%s",
        model_name,
        len(df),
        num_gpu,
    )
    run.start_time = time.time()

    for idx, row in df.iterrows():
        record_id = row["id"]
        text = row["text"]
        true_label = row["true_label"]

        predicted, raw_resp, latency, meta = classify_message(
            model_name=model_name,
            message_text=text,
            num_gpu=num_gpu,
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
