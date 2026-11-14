"""
MarkerPhish — Optional per-model tokenizer calibration
=======================================================
    python -m backend.calibrate <model> [--dataset seed+sms] [--safety 0.95]

Measures how many characters one token of the model holds on the platform's
densest prompts, and stores it as the model's chars_per_token override
(backend.model_overrides). The context check before a run then uses this value
instead of the conservative global estimate.

What it sends: at most 3 prefill-only requests (num_predict 1), the fewshotcot
prompt of (1) the longest email, (2) the email with the longest Links header
and (3) the longest SMS of the dataset choice. chars_per_token = the smallest
measured value x safety factor (default 0.95). The model is unloaded at the end.

Calibration is optional: a model without it uses the global estimate, and the
app only shows a note.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from config import (
    CHARS_PER_TOKEN_ESTIMATE,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_NUM_CTX,
    OLLAMA_SEED,
    build_classification_prompt,
)

from .manifest import model_details, now_iso
from .model_overrides import save_override

CALIBRATION_VARIANT = "fewshotcot"
DEFAULT_SAFETY = 0.95


def calibration_cases(df: pd.DataFrame) -> list[dict]:
    """Up to three worst-case records: longest email, longest Links header, longest SMS."""
    cases = []
    emails = df[df["channel"] == "email"]
    if not emails.empty:
        longest = emails.loc[emails["text"].str.len().idxmax()]
        cases.append({"case": "longest email", "row": longest})
        links = emails["text"].map(lambda t: len(t.split("\n")[2]) if t.count("\n") >= 2 else 0)
        densest = emails.loc[links.idxmax()]
        if densest["source_id"] != longest["source_id"]:
            cases.append({"case": "longest Links header", "row": densest})
    sms = df[df["channel"] == "sms"]
    if not sms.empty:
        cases.append({"case": "longest sms", "row": sms.loc[sms["text"].str.len().idxmax()]})
    return cases


def chars_per_token_from(measurements: list[dict], safety: float = DEFAULT_SAFETY) -> float | None:
    values = [m["chars_per_token"] for m in measurements if m.get("chars_per_token")]
    return round(min(values) * safety, 3) if values else None


def measure(model_name: str, cases: list[dict], num_ctx: int) -> list[dict]:
    out = []
    for case in cases:
        row = case["row"]
        prompt = build_classification_prompt(row["text"], row["channel"], CALIBRATION_VARIANT)
        resp = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "seed": OLLAMA_SEED, "num_ctx": num_ctx, "num_predict": 1},
            },
            timeout=1800,
        )
        resp.raise_for_status()
        tokens = resp.json().get("prompt_eval_count")
        out.append(
            {
                "case": case["case"],
                "source_id": str(row["source_id"]),
                "prompt_chars": len(prompt),
                "prompt_tokens": tokens,
                # a count at num_ctx means the prompt was cut, so the ratio is only a lower bound
                "prompt_cut": bool(tokens and tokens >= num_ctx - 1),
                "chars_per_token": round(len(prompt) / tokens, 4) if tokens else None,
            }
        )
    return out


def calibrate(model_name: str, dataset_choice: str = "seed+sms", safety: float = DEFAULT_SAFETY) -> dict:
    """Measure, store and return the model's override entry."""
    from .data_loader import load_dataset_with_report
    from .evaluator import unload_model

    meta = model_details(model_name)
    if meta.get("digest") is None:
        raise SystemExit(f"Model '{model_name}' is not installed in Ollama (see `ollama list`).")
    df, _ = load_dataset_with_report(dataset_choice)
    context = meta.get("context_length")
    num_ctx = max(OLLAMA_NUM_CTX, 8192)
    if context:
        num_ctx = min(num_ctx, int(context))
    try:
        measurements = measure(model_name, calibration_cases(df), num_ctx)
    finally:
        unload_model(model_name)
    cpt = chars_per_token_from(measurements, safety)
    if cpt is None:
        raise SystemExit("Ollama returned no prompt_eval_count; nothing was stored.")
    return save_override(
        model_name,
        chars_per_token=cpt,
        digest=meta.get("digest"),
        calibrated_at=now_iso(),
        measurements=measurements,
        note=f"min measured chars/token x {safety}; prompts: {CALIBRATION_VARIANT}, dataset {dataset_choice}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional per-model tokenizer calibration (max 3 requests).")
    parser.add_argument("model", help="model name as listed by `ollama list`")
    parser.add_argument("--dataset", default="seed+sms", help="dataset choice (default: seed+sms)")
    parser.add_argument("--safety", type=float, default=DEFAULT_SAFETY, help="factor applied to the measured value")
    args = parser.parse_args(argv)
    entry = calibrate(args.model, args.dataset, args.safety)
    for m in entry["measurements"]:
        print(f"{m['case']:22s} {m['source_id']:>14s}  {m['prompt_chars']:>6d} chars  "
              f"{m['prompt_tokens']} tokens  {m['chars_per_token']} chars/token")
    print(f"stored chars_per_token = {entry['chars_per_token']} for {args.model} "
          f"(global default: {CHARS_PER_TOKEN_ESTIMATE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
