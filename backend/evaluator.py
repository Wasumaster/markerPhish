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
import psutil
import requests

from config import (
    ACTIVE_PROMPT_VARIANT,
    COLD_UNLOAD_TIMEOUT_S,
    DEFAULT_START_MODE,
    DEFAULT_UNLOAD_OTHER_MODELS,
    START_MODES,
    DEFAULT_NUM_GPU,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_SEED,
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
from .hardware import hardware_block
from .manifest import model_details, now_iso, ollama_version, power_state
from .options import resolve_options, resolved_from_runner
from .profiler import ResourceSampler, model_matches, ollama_ps, read_tree_rss_mb, runner_cmdlines
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
    source_id: str = ""
    group_id: str = ""
    is_cold_request: bool = False  # cold start: the first request, which paid the model load
    load_duration_s: float = 0.0
    request_status: str = "ok"  # ok | timeout | error
    truncated_output: bool = False  # output hit num_predict: not parsed, counted as unknown
    input_possibly_truncated: bool = False  # prompt_eval_count reached num_ctx - num_predict
    prompt_eval_duration_s: float | None = None  # Ollama's own timings (seconds)
    eval_duration_s: float | None = None
    total_duration_s: float | None = None


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
    started_at: str = ""  # ISO 8601 with timezone, before the warm-up
    ended_at: str = ""
    # Ollama options of the evaluation requests, exactly as sent (first request);
    # requests that differed (GPU -> CPU fallback) are counted in options_changed.
    options_sent: dict = field(default_factory=dict)
    options_changed: int = 0
    request_fields: dict = field(default_factory=dict)  # format schema, think, stream
    power_at_start: dict | None = None
    power_at_end: dict | None = None
    power_changed: bool = False
    on_battery_during_run: bool = False
    start_mode: str = DEFAULT_START_MODE
    prerun: dict = field(default_factory=dict)  # loaded models before the run, unloads
    cold_unload_confirmed: bool | None = None
    pre_warmup_ram_mb: float | None = None
    baseline_ram_mb: float | None = None
    resources: dict = field(default_factory=dict)  # ResourceSampler summary
    hardware: dict = field(default_factory=dict)  # hardware block at run start
    options_resolution: dict = field(default_factory=dict)  # backend.options.ResolvedOptions.to_manifest()
    ollama_resolved: dict = field(default_factory=dict)  # num_thread / num_batch Ollama used, if readable

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


def find_model_size(models: list[dict], model_name: str) -> int:
    """
    Size in bytes of an installed model, matched by exact name.

    Ollama lists untagged models as "<name>:latest", so "mymodel" also matches
    "mymodel:latest". A prefix match is never used: "mymodel" must not pick up
    "mymodel:7b" when "mymodel:1.5b" is meant. Returns 0 when there is no match.
    """
    wanted = {model_name}
    if ":" not in model_name:
        wanted.add(model_name + ":latest")
    for m in models:
        if m.get("name") in wanted:
            return int(m.get("size") or 0)
    return 0


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


def _build_options(num_gpu: int | None, num_predict: int = DEFAULT_NUM_PREDICT) -> dict:
    """Ollama generation options shared by warm-up and evaluation requests."""
    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": num_predict,
        "seed": OLLAMA_SEED,
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


def warm_up_model(model_name: str, num_gpu: int | None = DEFAULT_NUM_GPU, options: dict | None = None) -> float:
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
        "options": {**(options or _build_options(num_gpu)), "num_predict": 1},
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
    options: dict | None = None,
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
        "options": dict(options) if options else _build_options(num_gpu),
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
        return "unknown", "[TIMEOUT]", elapsed, {"request_status": "timeout"}
    except requests.ConnectionError:
        elapsed = time.perf_counter() - start
        logger.error("Cannot connect to Ollama — is it running?")
        return "unknown", "[CONNECTION_ERROR: Ollama offline]", elapsed, {"request_status": "error"}
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
        return "unknown", f"[ERROR: {error_msg}]", elapsed, {"request_status": "error"}

    latency = time.perf_counter() - start
    raw_response = data.get("response", "").strip()
    sent_options = payload["options"]
    # An answer cut off at num_predict is never parsed: it counts as "unknown".
    truncated_output = data.get("done_reason") == "length"
    binary = "unknown" if truncated_output else parse_binary_label(raw_response)
    predicted = binary_to_channel_label(binary, channel)
    # Ollama shortens a prompt that does not fit num_ctx; prompt_eval_count counts the
    # whole prompt (also when it is cached), so reaching num_ctx - num_predict means
    # the input may have been cut.
    prompt_tokens = data.get("prompt_eval_count") or 0
    num_ctx = int(sent_options.get("num_ctx") or 0)
    input_possibly_truncated = bool(num_ctx) and prompt_tokens + int(sent_options.get("num_predict") or 0) >= num_ctx

    metadata = {
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "eval_count": data.get("eval_count", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "load_duration_ns": data.get("load_duration", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "request_status": "ok",
        "done_reason": data.get("done_reason"),
        "truncated_output": truncated_output,
        "input_possibly_truncated": input_possibly_truncated,
        "binary_label": binary,
        "channel": channel or "unknown",
        "prompt_variant": variant,
        # Options as finally sent (a GPU failure retries with num_gpu=0)
        "options_sent": dict(payload["options"]),
        "think_sent": "think" in payload,
    }

    return predicted, raw_response, latency, metadata


def _ns_to_s(value) -> float | None:
    return round(value / 1e9, 6) if isinstance(value, (int, float)) and value > 0 else None


def loaded_models() -> list[str] | None:
    """Names of the models currently loaded in Ollama (/api/ps), or None if unreachable."""
    models = ollama_ps()
    return None if models is None else [m.get("name") or m.get("model") for m in models]


def unload_model(model_name: str) -> bool:
    """Ask Ollama to unload a model now (keep_alive 0). Returns False on error."""
    try:
        resp = requests.post(
            OLLAMA_GENERATE_ENDPOINT, json={"model": model_name, "keep_alive": 0}, timeout=OLLAMA_REQUEST_TIMEOUT
        )
        return resp.status_code == 200
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not unload %s: %s", model_name, e)
        return False


def wait_until_unloaded(
    model_name: str,
    timeout_s: float = COLD_UNLOAD_TIMEOUT_S,
    poll_s: float = 0.5,
    sleep=time.sleep,
    clock=time.monotonic,
) -> bool:
    """Poll /api/ps until model_name is no longer loaded. Returns False on timeout."""
    deadline = clock() + timeout_s
    while True:
        models = ollama_ps()
        if models is not None and not any(model_matches(m, model_name) for m in models):
            return True
        if clock() >= deadline:
            return False
        sleep(poll_s)


def other_loaded_models(loaded: list[str] | None, model_name: str) -> list[str]:
    """Loaded models other than model_name ("x" and "x:latest" are the same model)."""
    return [m for m in (loaded or []) if not model_matches({"name": m}, model_name)]


def prepare_ollama(model_name: str, unload_others: bool = DEFAULT_UNLOAD_OTHER_MODELS) -> dict:
    """
    Pre-run check: which models are loaded; optionally unload every other one.
    Returns the facts stored in the run's resources block.
    """
    loaded = loaded_models()
    others = other_loaded_models(loaded, model_name)
    info = {
        "ollama_loaded_models": loaded,
        "other_models_loaded": bool(others),
        "other_models_unloaded": [],
        "other_models_unload_failed": [],
    }
    if others:
        logger.warning("Other models are loaded in Ollama before the run: %s", ", ".join(others))
        if unload_others:
            for name in others:
                ok = unload_model(name) and wait_until_unloaded(name)
                info["other_models_unloaded" if ok else "other_models_unload_failed"].append(name)
    return info


def power_changed(power_readings: list[bool | None]) -> tuple[bool, bool]:
    """
    (power_changed, on_battery_at_some_point) from AC readings taken during a run
    (start, sampler, end). Unknown readings (None) are ignored.
    """
    known = [p for p in power_readings if p is not None]
    return len(set(known)) > 1, (False in known)


def run_evaluation(
    model_name: str,
    df: pd.DataFrame,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    progress_callback=None,
    prompt_variant: str | None = None,
    start_mode: str = DEFAULT_START_MODE,
    unload_other_models: bool = DEFAULT_UNLOAD_OTHER_MODELS,
    sampler_factory=None,
    tree_rss_reader=read_tree_rss_mb,
    user_options: dict | None = None,
) -> EvaluationRun:
    """
    Run a full evaluation of a model against the provided dataset.

    Sequence:
      1. pre-run check: loaded models; unload the others (default)
      2. pre_warmup_ram_mb: Ollama process-tree RSS before the model is (re)loaded
      3. warm start: one excluded warm-up request (cold_start_s = its load time);
         cold start: unload the model and wait until /api/ps no longer lists it;
         the first evaluation request then pays the load (cold_start_s = its
         load_duration) and its latency is excluded from the latency statistics
      4. the resource sampler starts right before the first evaluation request
         (its first sample is baseline_ram_mb) and stops after the last one
      5. AC power is read at start, during sampling and at the end

    user_options (num_ctx, num_thread, num_predict, num_batch) are resolved and
    validated first (backend.options); an OptionsError is raised before anything
    is sent to Ollama when they are invalid or the longest prompt cannot fit.
    """
    if start_mode not in START_MODES:
        raise ValueError(f"Unknown start mode '{start_mode}'. Available: {', '.join(START_MODES)}")
    variant = prompt_variant or ACTIVE_PROMPT_VARIANT
    get_prompt_variant(variant)  # validate early
    run = EvaluationRun(model_name=model_name, total_records=len(df), prompt_variant=variant, start_mode=start_mode)

    models = list_available_models()
    run.model_size_bytes = find_model_size(models, model_name)

    model_meta = model_details(model_name)
    resolved = resolve_options(
        model_name=model_name,
        prompt_variant=variant,
        df_sample=df,
        model_context_length=model_meta.get("context_length"),
        logical_cores=psutil.cpu_count(logical=True) or 1,
        num_gpu=num_gpu,
        user=user_options,
    )
    options = resolved.options
    run.options_resolution = resolved.to_manifest()
    for line in resolved.notes:
        logger.info("Options: %s", line)

    logger.info(
        "Starting evaluation: model=%s, records=%d, num_gpu=%s, prompt_variant=%s, start=%s",
        model_name, len(df), num_gpu, variant, start_mode,
    )
    run.started_at = now_iso()
    run.power_at_start = power_state()
    run.hardware = hardware_block(ollama_version())
    run.request_fields = {
        "endpoint": "/api/generate",
        "stream": False,
        "format": build_output_schema(variant),
        "think": False if model_supports_thinking(model_name) else None,
    }

    run.prerun = prepare_ollama(model_name, unload_other_models)

    if start_mode == "cold":
        unload_model(model_name)
        run.cold_unload_confirmed = wait_until_unloaded(model_name, timeout_s=COLD_UNLOAD_TIMEOUT_S)
        if not run.cold_unload_confirmed:
            logger.warning("Cold start: %s was still loaded after %ss", model_name, COLD_UNLOAD_TIMEOUT_S)
        run.pre_warmup_ram_mb = tree_rss_reader()
    else:
        run.pre_warmup_ram_mb = tree_rss_reader()
        run.cold_start_s = warm_up_model(model_name, num_gpu, options)
        logger.info("Model warm-up finished (cold start %.2fs)", run.cold_start_s)

    sampler = sampler_factory(model_name) if sampler_factory else ResourceSampler(model_name)
    try:
        run.baseline_ram_mb = sampler.start()
    except Exception as e:  # noqa: BLE001 - profiling must never prevent the evaluation
        logger.warning("Resource sampler could not start: %s", e)
        sampler = None
    run.start_time = time.time()

    has_source = "source" in df.columns
    try:
        for _, row in df.iterrows():
            true_label = row["true_label"]
            channel = channel_for_source(row["source"] if has_source else None, true_label)

            predicted, raw_resp, latency, meta = classify_message(
                model_name=model_name,
                message_text=row["text"],
                num_gpu=num_gpu,
                channel=channel,
                prompt_variant=variant,
                options=options,
            )
            if not run.results:
                run.ollama_resolved = resolved_from_runner(runner_cmdlines())

            is_cold = start_mode == "cold" and not run.results
            load_s = (meta.get("load_duration_ns") or 0) / 1e9
            if is_cold:
                run.cold_start_s = load_s
            result = InferenceResult(
                record_id=row["id"],
                true_label=true_label,
                predicted_label=predicted,
                raw_response=raw_resp,
                latency_seconds=latency,
                prompt_eval_count=meta.get("prompt_eval_count", 0),
                eval_count=meta.get("eval_count", 0),
                is_binary_correct=is_binary_match(true_label, predicted),
                true_binary=to_binary_label(true_label),
                predicted_binary=to_binary_label(predicted),
                channel=channel,
                source_id=str(row["source_id"]) if "source_id" in df.columns else "",
                group_id=str(row["group_id"]) if "group_id" in df.columns else "",
                is_cold_request=is_cold,
                load_duration_s=round(load_s, 4),
                request_status=meta.get("request_status", "ok"),
                truncated_output=bool(meta.get("truncated_output")),
                input_possibly_truncated=bool(meta.get("input_possibly_truncated")),
                prompt_eval_duration_s=_ns_to_s(meta.get("prompt_eval_duration_ns")),
                eval_duration_s=_ns_to_s(meta.get("eval_duration_ns")),
                total_duration_s=_ns_to_s(meta.get("total_duration_ns")),
            )

            sent = meta.get("options_sent")
            if sent:
                if not run.options_sent:
                    run.options_sent = sent
                elif sent != run.options_sent:
                    run.options_changed += 1

            run.results.append(result)
            run.completed += 1
            if result.request_status != "ok":
                run.failed += 1

            if progress_callback:
                progress_callback(run.completed, run.total_records, result)

            if run.completed % 25 == 0:
                gc.collect()
    finally:
        run.end_time = time.time()
        if sampler is not None:
            try:
                run.resources = sampler.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("Resource sampler could not stop cleanly: %s", e)
        run.ended_at = now_iso()
        run.power_at_end = power_state()

    readings = [(run.power_at_start or {}).get("on_ac_power")]
    if run.resources.get("ac_power_lost_during_run"):
        readings.append(False)
    readings.append((run.power_at_end or {}).get("on_ac_power"))
    run.power_changed, run.on_battery_during_run = power_changed(readings)
    if run.power_changed or run.on_battery_during_run:
        logger.warning("Power: AC power was not present for the whole run (changed=%s)", run.power_changed)

    if not run.options_sent:  # every request failed before a response: record what was configured
        run.options_sent = dict(options)
    logger.info(
        "Evaluation complete: %d/%d processed (%.1fs, avg %.2fs/msg, failed=%d, peak RSS %s MB, model %s MB)",
        run.completed, run.total_records, run.elapsed_time, run.avg_latency, run.failed,
        run.resources.get("peak_ram_mb"), run.resources.get("model_size_mb"),
    )
    return run
