"""
MarkerPhish — Batch runner (headless)
======================================
    python -m backend.batch <experiment.yaml> [--dry-run]

Runs the full matrix of an experiment file (backend.experiment) without the UI:
one run JSON per cell (same format as the Run tab) plus a batch summary in
<results>/batches/batch_<batch_id>.json and a log in batch_<batch_id>.log.

- Before starting it prints the number of cells, how many are already done, and
  the estimated time (measured latencies of the same model + prompt variant from
  earlier runs where available; otherwise the estimate is labelled rough).
- It warns when the machine is on battery power (CPU latency depends on it).
- Resumable: a cell whose run file already exists and is complete is skipped.
- A failed cell is logged and the batch continues with the next one.
- Other loaded models are unloaded before every cell (unload_other_models), and
  the start mode (warm / cold) is applied to every cell.
- Power state is logged before and after every cell.
--dry-run prints the plan and sends no inference request.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

from config import DEFAULT_NUM_GPU

from . import utils
from .data_loader import load_dataset_with_report
from .evaluator import check_ollama_connection, list_available_models, run_evaluation
from .experiment import (
    ExperimentError,
    batch_id,
    estimate_seconds,
    expand_cells,
    load_experiment,
    measured_seconds_per_record,
    resolve_models,
)
from .manifest import now_iso, power_state
from .run_report import build_run_report
from .sampling import balanced_sample
from .utils import ensure_report_compatibility, load_result_by_filename, logger, save_results


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def batch_dir() -> str:
    return utils.ensure_dir(os.path.join(utils.RESULTS_DIR, "batches"))


def _saved_reports() -> dict[str, dict]:
    out = {}
    for name in utils.list_saved_result_files():
        rep = load_result_by_filename(name)
        if rep:
            out[name] = rep
    return out


def completed_cells(reports: dict[str, dict], bid: str, sample_size: int) -> dict[str, str]:
    """cell_id -> run file of complete runs of this batch."""
    done = {}
    for name, rep in reports.items():
        meta = rep.get("batch") or {}
        if meta.get("batch_id") != bid or not meta.get("cell_id"):
            continue
        n = ((rep.get("manifest") or {}).get("dataset") or {}).get("sample_size")
        if n == sample_size and len(rep.get("raw_predictions") or []) == sample_size:
            done[meta["cell_id"]] = name
    return done


def _write_summary(path: str, summary: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _power_text(state: dict | None) -> str:
    state = state or {}
    ac = state.get("on_ac_power")
    plan = (state.get("power_plan") or {}).get("name")
    return f"AC={'unknown' if ac is None else ac} battery={state.get('battery_percent')} plan={plan}"


def run_batch(path: str, dry_run: bool = False, sampler_factory=None, out=print) -> dict:
    """Run (or plan, with dry_run) the experiment at path. Returns the batch summary."""
    experiment = load_experiment(path)
    if not check_ollama_connection():
        raise ExperimentError("Ollama is not reachable. Start it (ollama serve) and try again.")
    installed = [m["name"] for m in list_available_models()]
    models = resolve_models(experiment, installed)
    bid = batch_id(experiment, models)
    cells = expand_cells(experiment, models)

    df, _ = load_dataset_with_report(experiment.dataset)
    sample = balanced_sample(df, experiment.sample_size, seed=experiment.seed)
    if sample.info["fallback"]:
        out(f"WARNING: {sample.info['fallback_reason']}")
    sample_size = sample.info["actual_size"]

    reports = _saved_reports()
    done = completed_cells(reports, bid, sample_size)
    todo = [c for c in cells if c.cell_id not in done]
    measured = measured_seconds_per_record([ensure_report_compatibility(r) for r in reports.values()])
    seconds, quality = estimate_seconds(todo, sample_size, measured)

    out(f"Experiment '{experiment.name}' - batch {bid}")
    out(f"Models: {', '.join(models)}")
    out(f"Cells: {len(cells)} ({len(done)} already complete, {len(todo)} to run) - {sample_size} records each")
    out(f"Estimated time: {_fmt_duration(seconds)} ({quality} estimate"
        + ("; no earlier runs of these model/prompt pairs" if quality == "rough" else "") + ")")
    power = power_state()
    if power.get("on_ac_power") is False:
        out("WARNING: the machine is on battery power. CPU latency depends on the power source; "
            "plug in AC power for comparable results.")
    if dry_run:
        for c in cells:
            out(f"  [{'done' if c.cell_id in done else 'todo'}] {c.index + 1:>3}. {c.label()}")
        return {"batch_id": bid, "cells": len(cells), "todo": len(todo), "estimate_s": seconds, "dry_run": True}

    summary_path = os.path.join(batch_dir(), f"batch_{bid}.json")
    log_path = os.path.join(batch_dir(), f"batch_{bid}.log")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    summary = {
        "batch_id": bid,
        "experiment_file": os.path.abspath(path),
        "definition": experiment.definition(models),
        "sample": {k: sample.info[k] for k in ("requested_size", "actual_size", "fallback", "sample_seed", "cells")},
        "started_at": now_iso(),
        "ended_at": None,
        "power_at_start": power,
        "cells": [
            {"cell_id": c.cell_id, "index": c.index, "model": c.model, "prompt_variant": c.prompt_variant,
             "options": c.options, "repetition": c.repetition,
             "status": "complete (earlier)" if c.cell_id in done else "pending",
             "run_file": done.get(c.cell_id)}
            for c in cells
        ],
    }
    _write_summary(summary_path, summary)
    logger.info("Batch %s started: %d cells to run, estimate %s (%s)", bid, len(todo), _fmt_duration(seconds), quality)

    batch_start = time.monotonic()
    records_done = 0
    total_records = len(todo) * sample_size
    try:
        for n_cell, cell in enumerate(todo, start=1):
            entry = summary["cells"][cell.index]
            before = power_state()
            logger.info("Cell %d/%d: %s - power before: %s", n_cell, len(todo), cell.label(), _power_text(before))
            entry.update(status="running", started_at=now_iso(), power_before=before)
            _write_summary(summary_path, summary)
            cell_start = time.monotonic()
            step = max(1, sample_size // 10)

            def progress(current, total, result, _start=cell_start, _n=n_cell):
                if current % step == 0 or current == total:
                    elapsed = time.monotonic() - batch_start
                    finished = records_done + current
                    eta = elapsed / finished * (total_records - finished) if finished else 0
                    logger.info("  cell %d/%d: %d/%d records - batch ETA %s",
                                _n, len(todo), current, total, _fmt_duration(eta))

            try:
                run = run_evaluation(
                    model_name=cell.model,
                    df=sample.df,
                    num_gpu=DEFAULT_NUM_GPU,
                    progress_callback=progress,
                    prompt_variant=cell.prompt_variant,
                    start_mode=experiment.start_mode,
                    unload_other_models=experiment.unload_other_models,
                    user_options=cell.options,
                    **({"sampler_factory": sampler_factory} if sampler_factory else {}),
                )
                report = build_run_report(
                    run,
                    df_sample=sample.df,
                    dataset_choice=experiment.dataset,
                    sample_info=sample.info,
                    execution_mode="CPU" if DEFAULT_NUM_GPU == 0 else "GPU",
                )
                report["batch"] = {
                    "batch_id": bid,
                    "cell_id": cell.cell_id,
                    "cell_index": cell.index,
                    "repetition": cell.repetition,
                    "experiment": experiment.name,
                    "constraint_profiles": experiment.constraint_profiles,
                }
                run_file = os.path.basename(save_results(report, cell.model, prompt_variant=cell.prompt_variant))
                entry.update(status="complete", run_file=run_file)
                if run.power_changed or run.on_battery_during_run:
                    logger.warning("  cell %d: AC power was not present for the whole run", n_cell)
            except Exception as e:  # noqa: BLE001 - one failed cell must not stop the batch
                entry.update(status="failed", error=f"{type(e).__name__}: {e}")
                logger.error("Cell %d failed: %s\n%s", n_cell, e, traceback.format_exc())
            after = power_state()
            entry.update(ended_at=now_iso(), duration_s=round(time.monotonic() - cell_start, 1), power_after=after)
            logger.info("Cell %d/%d %s in %s - power after: %s", n_cell, len(todo), entry["status"],
                        _fmt_duration(entry["duration_s"]), _power_text(after))
            records_done += sample_size
            _write_summary(summary_path, summary)
    except KeyboardInterrupt:
        logger.warning("Batch interrupted; restart the same command to resume.")
        for entry in summary["cells"]:
            if entry["status"] == "running":
                entry["status"] = "interrupted"
        raise
    finally:
        summary["ended_at"] = now_iso()
        _write_summary(summary_path, summary)
        logger.removeHandler(handler)
        handler.close()

    counts = {}
    for entry in summary["cells"]:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    out(f"Batch {bid} finished: {counts}. Summary: {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an experiment matrix headless (see backend/experiment.py).")
    parser.add_argument("experiment", help="path to the experiment YAML file")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, run nothing")
    args = parser.parse_args(argv)
    try:
        run_batch(args.experiment, dry_run=args.dry_run)
    except ExperimentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
