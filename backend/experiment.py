"""
MarkerPhish — Experiment definitions (batch mode)
==================================================
An experiment file (YAML) describes a matrix of runs:

    name: my-experiment              # free text
    models: installed                # "installed" = every model Ollama lists, or a list of names
    prompt_variants: [zeroshot, fewshot, fewshotcot]   # or "all"
    dataset: seed+sms                # a value of data_loader.DATASET_CHOICES
    sample_size: 250                 # records per run (>= 20)
    seed: 42                         # sample seed (default config.SAMPLE_SEED)
    repetitions: 1                   # runs per cell
    start_mode: warm                 # warm | cold
    unload_other_models: true
    constraint_profiles: [laptop-8gb, laptop-16gb]   # recorded with the batch
    options:                         # optional Ollama options; a list expands the matrix
      num_thread: [4, 8]
      num_ctx: 4096

Cells = models x prompt variants x every combination of list-valued options x
repetitions. Several tags of one model (e.g. two quantizations) are simply
several entries in `models`.

The batch id is a hash of the definition (with the resolved model list), and
the cell id a hash of everything that defines one run, so a restarted batch
recognises the cells it already finished.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

import yaml

from config import (
    CONSTRAINT_PROFILES,
    DEFAULT_CONSTRAINT_PROFILES,
    DEFAULT_START_MODE,
    DEFAULT_UNLOAD_OTHER_MODELS,
    MIN_SAMPLE_SIZE,
    PROMPT_VARIANTS,
    SAMPLE_SEED,
    START_MODES,
)

from .options import CONFIGURABLE

ALLOWED_KEYS = {
    "name", "models", "prompt_variants", "dataset", "sample_size", "seed", "repetitions",
    "start_mode", "unload_other_models", "constraint_profiles", "options", "description",
}
INSTALLED = "installed"
# Fallback seconds per record for a rough estimate when nothing was measured yet.
ROUGH_SECONDS_PER_RECORD = 5.0


class ExperimentError(ValueError):
    """Invalid experiment definition; the message names the problem."""


@dataclass
class Experiment:
    name: str
    models: list[str] | str
    prompt_variants: list[str]
    dataset: str
    sample_size: int
    seed: int = SAMPLE_SEED
    repetitions: int = 1
    start_mode: str = DEFAULT_START_MODE
    unload_other_models: bool = DEFAULT_UNLOAD_OTHER_MODELS
    constraint_profiles: list[str] = field(default_factory=lambda: list(DEFAULT_CONSTRAINT_PROFILES))
    options: dict = field(default_factory=dict)
    source_path: str | None = None

    def definition(self, resolved_models: list[str]) -> dict:
        return {
            "name": self.name,
            "models": resolved_models,
            "prompt_variants": self.prompt_variants,
            "dataset": self.dataset,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "start_mode": self.start_mode,
            "unload_other_models": self.unload_other_models,
            "constraint_profiles": self.constraint_profiles,
            "options": self.options,
        }


@dataclass(frozen=True)
class Cell:
    index: int
    model: str
    prompt_variant: str
    options: dict
    repetition: int
    cell_id: str

    def label(self) -> str:
        opts = ", ".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{self.model} | {self.prompt_variant}" + (f" | {opts}" if opts else "") + f" | rep {self.repetition}"


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def parse_experiment(data: dict, source_path: str | None = None) -> Experiment:
    """Validate a parsed YAML/JSON mapping and return an Experiment."""
    from .data_loader import DATASET_CHOICES

    if not isinstance(data, dict):
        raise ExperimentError("The experiment file must contain a mapping (key: value lines).")
    unknown = set(data) - ALLOWED_KEYS
    if unknown:
        raise ExperimentError(f"Unknown keys: {', '.join(sorted(unknown))}")
    for key in ("models", "prompt_variants", "dataset", "sample_size"):
        if key not in data:
            raise ExperimentError(f"Missing required key: {key}")

    models = data["models"]
    if isinstance(models, str):
        if models != INSTALLED:
            models = [models]
    elif not (isinstance(models, list) and models and all(isinstance(m, str) and m for m in models)):
        raise ExperimentError("models must be 'installed' or a non-empty list of model names")

    variants = data["prompt_variants"]
    if variants == "all":
        variants = list(PROMPT_VARIANTS)
    if isinstance(variants, str):
        variants = [variants]
    bad = [v for v in variants if v not in PROMPT_VARIANTS]
    if bad or not variants:
        raise ExperimentError(f"Unknown prompt variants: {bad}. Available: {', '.join(PROMPT_VARIANTS)}")

    if data["dataset"] not in DATASET_CHOICES.values():
        raise ExperimentError(f"Unknown dataset '{data['dataset']}'. Available: {', '.join(DATASET_CHOICES.values())}")

    def integer(key, default, minimum):
        value = data.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ExperimentError(f"{key} must be a whole number >= {minimum}")
        return value

    sample_size = integer("sample_size", None, MIN_SAMPLE_SIZE)
    seed = integer("seed", SAMPLE_SEED, 0)
    repetitions = integer("repetitions", 1, 1)

    start_mode = data.get("start_mode", DEFAULT_START_MODE)
    if start_mode not in START_MODES:
        raise ExperimentError(f"start_mode must be one of {', '.join(START_MODES)}")
    unload = data.get("unload_other_models", DEFAULT_UNLOAD_OTHER_MODELS)
    if not isinstance(unload, bool):
        raise ExperimentError("unload_other_models must be true or false")

    profiles = data.get("constraint_profiles", list(DEFAULT_CONSTRAINT_PROFILES))
    if isinstance(profiles, str):
        profiles = [profiles]
    missing = [p for p in profiles if p not in CONSTRAINT_PROFILES]
    if missing:
        raise ExperimentError(f"Unknown constraint profiles: {missing}. Available: {', '.join(CONSTRAINT_PROFILES)}")

    options = data.get("options") or {}
    if not isinstance(options, dict):
        raise ExperimentError("options must be a mapping")
    bad_opts = set(options) - set(CONFIGURABLE)
    if bad_opts:
        raise ExperimentError(f"Not configurable: {', '.join(sorted(bad_opts))}. Configurable: {', '.join(CONFIGURABLE)}")
    for key, value in options.items():
        values = value if isinstance(value, list) else [value]
        if not values or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in values):
            raise ExperimentError(f"options.{key} must be a whole number >= 1 or a list of them")

    return Experiment(
        name=str(data.get("name") or "experiment"),
        models=models,
        prompt_variants=list(variants),
        dataset=data["dataset"],
        sample_size=sample_size,
        seed=seed,
        repetitions=repetitions,
        start_mode=start_mode,
        unload_other_models=unload,
        constraint_profiles=list(profiles),
        options=options,
        source_path=source_path,
    )


def load_experiment(path: str) -> Experiment:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise ExperimentError(f"Cannot read {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ExperimentError(f"{path} is not valid YAML: {e}") from e
    return parse_experiment(data, source_path=path)


def resolve_models(experiment: Experiment, installed: list[str]) -> list[str]:
    """The experiment's model list against what Ollama has installed (x == x:latest)."""
    if experiment.models == INSTALLED:
        if not installed:
            raise ExperimentError("models: installed, but Ollama lists no models")
        return list(installed)
    resolved, missing = [], []
    for name in experiment.models:
        match = next((m for m in installed if m == name or m == f"{name}:latest"), None)
        (resolved if match else missing).append(match or name)
    if missing:
        raise ExperimentError(f"Not installed in Ollama: {', '.join(missing)}")
    return resolved


def option_grid(options: dict) -> list[dict]:
    """Every combination of list-valued options (scalars stay fixed)."""
    keys = sorted(options)
    values = [options[k] if isinstance(options[k], list) else [options[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)] if keys else [{}]


def batch_id(experiment: Experiment, resolved_models: list[str]) -> str:
    return _hash(experiment.definition(resolved_models))[:10]


def expand_cells(experiment: Experiment, resolved_models: list[str]) -> list[Cell]:
    """The run matrix in execution order: model -> variant -> options -> repetition."""
    cells = []
    for model in resolved_models:
        for variant in experiment.prompt_variants:
            for opts in option_grid(experiment.options):
                for rep in range(1, experiment.repetitions + 1):
                    key = {
                        "dataset": experiment.dataset,
                        "sample_size": experiment.sample_size,
                        "seed": experiment.seed,
                        "start_mode": experiment.start_mode,
                        "model": model,
                        "prompt_variant": variant,
                        "options": opts,
                        "repetition": rep,
                    }
                    cells.append(Cell(len(cells), model, variant, opts, rep, _hash(key)[:12]))
    return cells


def estimate_seconds(cells: list[Cell], sample_size: int, measured: dict) -> tuple[float, str]:
    """
    Estimated total time. measured: {(model, variant): seconds per record}, from
    earlier runs whose latency is not inflated. Cells without a measurement use
    the same model's other variants, else ROUGH_SECONDS_PER_RECORD.
    Returns (seconds, quality) with quality "measured", "partly rough" or "rough".
    """
    total = 0.0
    exact = rough = 0
    for cell in cells:
        per_record = measured.get((cell.model, cell.prompt_variant))
        if per_record is not None:
            exact += 1
        else:
            same_model = [v for (m, _), v in measured.items() if m == cell.model]
            per_record = max(same_model) if same_model else ROUGH_SECONDS_PER_RECORD
            rough += 1
        total += per_record * sample_size
    quality = "measured" if rough == 0 else ("rough" if exact == 0 else "partly rough")
    return total, quality


def measured_seconds_per_record(reports: list[dict]) -> dict:
    """{(model, variant): median of the per-run mean latency} from non-inflated, non-legacy runs."""
    import statistics

    from .manifest import report_latency_inflated, report_protocol_version

    per: dict = {}
    for rep in reports:
        if report_protocol_version(rep) < 2 or report_latency_inflated(rep):
            continue
        mean = ((rep.get("timing") or {}).get("latency_s") or {}).get("mean")
        if mean:
            key = (rep.get("model_name"), (rep.get("manifest") or {}).get("prompt", {}).get("prompt_variant"))
            per.setdefault(key, []).append(float(mean))
    return {k: statistics.median(v) for k, v in per.items()}


def estimate_run_seconds(model: str, prompt_variant: str, n_records: int, measured: dict) -> tuple[float, str]:
    """Estimated duration of one run (seconds, 'measured' | 'rough'), same rules as estimate_seconds."""
    cell = Cell(0, model, prompt_variant, {}, 1, "")
    seconds, quality = estimate_seconds([cell], n_records, measured)
    return seconds, quality


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")

