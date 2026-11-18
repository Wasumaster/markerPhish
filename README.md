# MarkerPhish

Platform for evaluating language models in phishing and smishing detection with a focus on resource efficiency.

MarkerPhish runs local language models (served by [Ollama](https://ollama.com)) on public phishing email and
smishing SMS datasets, and measures, for every run:

- detection quality: precision, recall, F1, FPR and FNR for the malicious class, overall and per channel, with
  bootstrap confidence intervals, in two treatments of unanswered messages;
- cost: latency per message, throughput, tokens per second, model memory, process memory and CPU over time;
- whether the configuration fits a resource-constrained deployment (a laptop with 8 or 16 GB RAM and no
  dedicated GPU), and which configuration is the best choice per profile, with paired significance tests.

Every run is saved as a JSON file with a reproducibility manifest (sample, prompt, model digest, options,
software versions, power state). The methodology, formulas and the file format are in **[OVERVIEW.md](OVERVIEW.md)**.

## Setup

Requirements: Windows 10/11, Linux or macOS; Python 3.13 (results were produced with 3.13.3; the code needs at
least 3.10); [Ollama](https://ollama.com) running locally; at least one model pulled.

```powershell
git clone https://github.com/Wasumaster/markerPhish.git
cd markerPhish
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # exact, pinned versions
pip install -r requirements-dev.txt      # optional: pytest, to run the tests
ollama pull <model>                      # any model; the app lists whatever Ollama has installed
```

Run the tests with `python -m pytest` (they use a fake Ollama and never load a model).

## Run one evaluation

Start the app with `run.bat` (Windows) or `streamlit run app.py`, then open `http://localhost:8501`.

1. **Sidebar:** pick a model, a prompt variant (zeroshot / fewshot / fewshotcot), the dataset (default
   PhishFuzzer Seed + SMS) and the sample size (50, 100, 250, 500, 1000 or custom; 50 and 100 are smoke tests,
   sizes below 500 are for testing and the pilot only).
2. **Dataset:** the sample and the preprocessing report (spam removed, duplicates, truncation, records per cell).
3. **Run:** check the configuration (per-cell counts, estimated time), the start mode (warm / cold), the
   loaded-models warning, optionally the "advanced" options (device CPU or GPU, default CPU; inference
   options), then **Run evaluation**. The
   "non-LLM baseline" expander trains and evaluates a TF-IDF + logistic regression reference on the same sample
   in seconds. It is shown in Compare as a reference point but is excluded from the recommendation.
4. **Results:** detection with confidence intervals, per channel, cost, resources over time, hardware and manifest.
5. **Compare:** all runs side by side, the recommendation per constraint profile, the Pareto chart and pairwise
   McNemar tests. A dataset filter (default: the dataset of the most recent run) and a dataset column keep
   runs on different data apart; if the selected runs differ in dataset or sample, a warning names them and
   they are never bold and never recommended.

Optional: `python -m backend.calibrate <model>` measures the model's tokenizer density (at most three requests)
for the context check. Without it the app uses a conservative default and shows a note.

## Batch mode (headless, many runs)

An experiment file (YAML; format and example in [OVERVIEW.md §12](OVERVIEW.md#12-batch-mode)) describes a matrix of
models × prompt variants × options × repetitions. Run it without the UI:

```powershell
python -m backend.batch path\to\experiment.yaml --dry-run   # plan and time estimate, sends nothing
python -m backend.batch path\to\experiment.yaml             # run; restart the same command to resume
```

One run JSON per cell lands in `results/` (the same files the UI reads); the batch summary and log are in
`results/batches/`. Finished cells are skipped on restart; a failed cell is logged and the batch continues.

**Running overnight on Windows**

- Plug in AC power (the batch warns on battery; CPU clocks drop on battery) and keep the power plan fixed.
- Prevent sleep while plugged in: `powercfg /change standby-timeout-ac 0` and `powercfg /change hibernate-timeout-ac 0`.
- If the lid will be closed: `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` then
  `powercfg /setactive SCHEME_CURRENT` (closing the lid then does nothing on AC). Otherwise keep the lid open.
- Close other heavy programs; the app itself need not run. Pause OneDrive sync, or move the results out of a
  synced folder with the environment variable `MARKERPHISH_RESULTS_DIR`.
- Start the batch in a terminal that stays open.

## Repository layout

| path | content |
|---|---|
| `app.py`, `frontend/`, `ui/` | Streamlit UI and design system |
| `backend/` | data pipeline, sampling, Ollama client, profiling, metrics, manifest, optimum analysis, batch, baseline |
| `config.py` | all constants (labels, prompt, context budget, profiles, protocol) |
| `Data/` | PhishFuzzer (email) and Mendeley (SMS) datasets |
| `results/` | saved runs (`eval_*.json`); `results/batches/` batch summaries and logs |
| `tests/` | pytest suite (fake Ollama) |
| `docs/` | audit, refactor task and progress log |

The six runs from October 2026 in `results/` use an older protocol (OVERVIEW.md §15) and have latencies inflated by
about 2 s per message (§13.3); Compare hides them by default.

## License & citation

Developed as part of an engineering thesis. The datasets keep their original licenses (see `Data/`).
