<!-- The refactor task as given on 2026-11-14 (verbatim). Source of the phase plan in REFACTOR_PROGRESS.md. -->

# MarkerPhish: full refactor based on the audit

This task implements the audit of MarkerPhish (weaknesses #1–23, recommendations #1–16) and the resource-profiling plan that was paused earlier.

The audit is saved in docs/AUDIT.md and is the source of truth; read it before starting. The profiling plan is summarized in Phase 4 below.

Create and maintain docs/REFACTOR_PROGRESS.md:
- a checklist of all phases and their items;
- updated at the end of every phase with what was done, deviations and open issues.
If this conversation is compacted or restarted, resume from that file.

Priority: Phases 1–6 change what the model sees and what each run records, so no run made before they are finished is usable for my thesis. After Phase 6, STOP and tell me explicitly that I can start the pilot experiment. Phases 7–10 work on stored data and will be done while the pilot runs. Items marked (optional) are done last, and only if I confirm.

## Thesis context (decides every trade-off)
- Title: "Platform for evaluating language models in phishing and smishing detection with a focus on resource efficiency".
- The description commits to:
  - automated, reproducible evaluation
  - preprocessed public datasets
  - models of various sizes
  - precision/recall/F1
  - concurrent measurement of latency, memory footprint and hardware requirements
  - a trade-off analysis that identifies an optimal solution for resource-constrained deployment

## My decisions
1. Reference hardware for all thesis results: this machine, Ryzen 7 5700U, integrated AMD GPU, no NVIDIA, CPU-only inference. GPU/VRAM code must work when a GPU exists, but nothing may depend on it.
2. "Resource-constrained environment" = a laptop with 8–16 GB RAM and no dedicated GPU. This is the default constraint profile (Phase 4).
3. Sample size is per run (one model × one prompt variant × one set of inference options), not per batch.
   - Selectable sizes: 50, 100, 250, 500, 1000, and custom (any integer ≥ 20). Default: 1000.
   - Sampling is balanced by class AND channel into four cells: phishing email, valid email, smishing SMS, valid SMS. It uses the fixed seed and is group-aware (Phase 2).
   - When the size is not divisible by 4 (e.g. 50, 250), use largest-remainder allocation: the extra records go to cells in a fixed, documented order. Store the exact per-cell counts in the manifest.
   - If a cell does not have enough unique records after deduplication, report it and fall back to the largest balanced size possible. Never silently unbalance.
   - Sizes below 500 are for testing and the pilot only. The UI labels 50 and 100 as "smoke test, not for results".
   - Repetitions: default 1, since results are deterministic (confirmed earlier). A separate small latency-variance experiment (100 records × 3 repetitions) covers timing variance.
4. Excluding spam from both datasets is a deliberate decision, agreed with my supervisor. Keep it, and document the rationale and its effect on FPR interpretation.
5. Quantization (Q4 vs Q8 of the same model) as an experiment axis: undecided. The platform must SUPPORT it (record quantization, allow several tags of one model in a batch, compare them), but do not make it mandatory or run such experiments.
6. Experiment design: a two-stage study.
   - Pilot: all installed models × 3 prompt variants on 250 records. Purpose: pick the best prompt variant per model and measure real latencies.
   - Main: each model with its chosen variant on 1000 records.
   - Latency variance: a small repeated experiment.

## Global rules (apply to every phase)
- Work in the phases below, in order. After each phase:
  1. run the tests;
  2. start the app and check every tab still loads (Streamlit AppTest);
  3. update docs/REFACTOR_PROGRESS.md;
  4. STOP and give me a short report: files changed, what was done, deviations from this prompt, open issues;
  5. wait for me to reply "continue". I commit between phases. Never commit yourself.
- Before Phase 1, give me a plan: phases, files per phase, anything in this prompt you think is wrong or risky, and questions. Wait for approval.
- Backward compatibility: every existing run JSON (including legacy ones) must still load. Introduce protocol_version:
  - legacy runs = 1, inferred on load;
  - runs produced after Phase 6 is complete = 2;
  - runs produced during the refactor (Phases 1–6) are tagged "dev": true. They are excluded from Compare by default and never enter the recommendation.
- Do not change the core of the prompt variants (zeroshot / fewshot / fewshotcot) beyond what Phase 2 requires. If a template must change, it changes for all variants consistently, and the prompt hash (Phase 3) records it.
- Temperature stays 0, and the fixed sample seed stays.
- UI follows the existing design system (ui/theme.py, ui/components.py): no st.metric, gradients, pills or emoji. Same tokens and table component.
- Keep the code modular: new logic goes in backend/ as pure, testable functions; the UI only renders. No business logic in frontend files.
- Every new feature gets unit tests in a real tests/ folder committed to the repo (pytest). Mock Ollama in tests; the unit test suite never requires a running model.
- If something in this prompt turns out to be impossible or a bad idea once you see the code, say so in the phase report instead of silently working around it.

## Phase 1: bugs, hygiene, test infrastructure
- Add a tests/ folder with pytest, a conftest.py and a shared Ollama mock.
- Cover the existing logic first, so later phases cannot break it unnoticed:
  - parse_binary_label
  - compute_classification_metrics
  - stratified_sample
  - channel_for_source
  - ensure_report_compatibility
- Fix the "select all" button on Compare. It must update the multiselect's own key, then rerun.
- Wrap the run in runner.py in try/finally, so that is_running is always reset after an error or an interrupted rerun.
- Use exact model-name matching instead of startswith when looking up model size, so qwen2.5:1.5b never matches qwen2.5:7b.
- Cache the dataset load with st.cache_data, keyed by dataset choice and preprocessing version.
- ensure_report_compatibility: do not silently assume true="valid" when the label is missing. Mark such records as unknown_ground_truth, exclude them from metrics and count them.
- Dependencies:
  - pin exact versions in requirements.txt, matching what is installed now (Streamlit 1.66+);
  - remove matplotlib and seaborn if they are truly unused;
  - add a short note in README on the Python version.
- Remove the dead code listed in the audit (BINARY_LABEL_MAP, DEFAULT_MODELS, EvaluationRun.to_dataframe, utils.load_latest_results, frontend/views/, the frontend/styles.py shim), but only after grepping that nothing imports it. List what you removed.

## Phase 2: data preprocessing (audit critical #1, important #7)
- PhishFuzzer emails: build the model input so the model sees the signals the prompt asks about:
  - "From: <sender>"
  - "Subject: …"
  - "Links: <URLs from the URL field>" (deduplicated)
  - then the body
  Choose one clear, fixed format and document it.
- SMS: unchanged except for cleaning.
- Truncation:
  - never drop the From / Subject / Links header;
  - truncate only the body, so that all links survive (they are already in the Links header);
  - choose the body limit so header + body + prompt fits num_ctx with margin, explain the number, and make it a config constant;
  - per record, store truncated (bool) and original_length;
  - report per dataset: how many records were truncated, and the median and max chars removed.
- Deduplication:
  - remove exact duplicates and normalized duplicates (lowercase, collapsed whitespace, stripped punctuation at the ends) within each source, before sampling;
  - if the same text appears with conflicting labels, drop all copies and count them separately;
  - report: records before, after, exact dupes, normalized dupes, label conflicts.
- Keep source identifiers on every record: PhishFuzzer No. / Original_ID, the Mendeley CSV row number. Keep the internal sequential id, but add source_id.
- Seed + rephrased ("all" combination):
  - add group_id = Original_ID (seed records use their own No.);
  - sampling is group-aware: a group is either fully in the sample or not, so near-identical rephrasings do not inflate the effective sample size;
  - report how many groups the sample contains.
- Implement the balanced, channel-stratified, group-aware sampling from decision 3, including the largest-remainder rule and the fallback. Check and report how many unique records each of the four cells has after deduplication, so I know the maximum balanced sample size per dataset choice.
- Spam exclusion: keep it, make it an explicit, documented config constant, and report how many records it removed per source.
- Add PREPROCESSING_VERSION (string) to config, and store it in every run (Phase 3).
- Show the dataset report (truncation, deduplication, spam removed, groups, available records per cell) on the Dataset tab, in the existing table style.

## Phase 3: reproducibility manifest (audit critical #5)
Every new run JSON gets a "manifest" block with:
- protocol_version, dev flag, preprocessing_version
- dataset_choice, sample_size, per-cell counts, sample_seed, group_aware (bool)
- the list of sampled source_ids, with group_ids where relevant
- dataset_fingerprint: hash of the sampled records' ids + texts + labels
- prompt_variant, prompt_template_hash (SHA-256 of the exact template text) and the full template text
- model: name, tag, digest, quantization, parameter_size, family, context length (from /api/show and /api/tags)
- inference options as actually sent (see Phase 5b), including temperature and seed (add seed to the Ollama options)
- Ollama version
- code version: git commit hash and a dirty flag. If git is unavailable, store null; never crash.
- timestamps (start/end, ISO 8601, with timezone)

Results tab:
- a collapsed "Manifest" expander;
- if the current code, model digest or preprocessing version differ from the run's, list the differences;
- (optional) a "Reproduce" button that pre-fills the Run tab with the same settings.

## Phase 4: resource profiling (audit critical #2)
Context from the earlier inspection of this machine:
- backend/hardware.py scan_hardware() already builds a HardwareProfile (CPU, cores, RAM, NVIDIA GPU via nvidia-smi). Extend it; do not duplicate it.
- warm_up_model() already sends one excluded warm-up request and stores cold_start_s.
- Ollama process tree on this machine:
  - "ollama app.exe" (tray app)
  - → "ollama.exe serve"
  - → "llama-server.exe --model …", the actual runner, which appears only when a model is loaded.
- /api/ps returns size and size_vram (size_vram = 0 here, CPU-only). The Ollama version comes from /api/version.
- psutil is installed; nvidia-ml-py is not.

Implement:
1. Process matching: find the root "ollama.exe" whose command line contains "serve" (fallback: any ollama* process that is not the tray app), plus all its descendants, recursively.
   - Re-resolve the tree on every sample, because the runner can spawn mid-run.
   - Match lowercase names and command lines, so macOS/Linux names also work.
   - Exclude "ollama app.exe".
2. A ResourceSampler running in a background daemon thread every 0.5 s. Each sample records:
   - process-tree RSS sum;
   - process-tree CPU %, normalized to the whole machine (0–100 %, divided by logical cores);
   - NVIDIA VRAM used and GPU utilization via pynvml, or null when unavailable, without errors or noisy warnings.
   It also polls /api/ps about every 2 s, keeping the peak size and size_vram, and times itself (sampler_cost_ms).
3. Primary memory metric: model memory from /api/ps. Secondary: process-tree RSS. Document that RSS is noisy (mmap, OS page cache).
4. Warm/cold start option, default warm:
   - warm: the existing warm_up_model(), excluded from all metrics;
   - cold: unload with keep_alive: 0, poll /api/ps until the model is gone (30 s timeout); the first request pays the load, and cold_start_s comes from its load_duration.
   The sampler starts just before the first evaluation request and stops after the last one.
5. Baseline RAM: sampled immediately before the first evaluation request. Also store pre_warmup_ram_mb, the idle server.
6. Pre-run check:
   - store ollama_loaded_models;
   - if any model other than the evaluated one is loaded, set other_models_loaded: true, log it, show a warning in the Run tab and offer to unload the other models (recommended default).
7. A "resources" block in the run JSON with explicit units:
   - peak_ram_mb, mean_ram_mb, baseline_ram_mb, pre_warmup_ram_mb
   - peak_cpu_pct, mean_cpu_pct
   - model_size_mb, model_vram_mb
   - peak_vram_mb, mean_vram_mb, mean_gpu_util_pct
   - sample_interval_s, n_samples, sampler_cost_ms, start_mode
   - ollama_loaded_models, other_models_loaded
   Plus "samples": a list of {t_s, ram_mb, cpu_pct, vram_mb}.
8. A "hardware" block, built by extending HardwareProfile:
   - CPU model, physical and logical cores, total RAM
   - GPU name and total VRAM (if any)
   - OS version, Python version, Ollama version
   Keep the old hardware_profile key for compatibility.
9. Constraint profiles are defined HERE, in config (Phase 7 only adds UI editing and the analysis):
   - "laptop-8gb": no GPU, model memory budget = 8 GB × a headroom factor;
   - "laptop-16gb": no GPU, model memory budget = 16 GB × the same factor;
   - optional latency limit (P95 ≤ X s), default none.
   Propose the headroom factor with a justification (OS and other apps need RAM), and document it.
   Add a derived "fits_profile" field per profile: model memory is the primary criterion, peak RSS is reported alongside.
10. Add nvidia-ml-py to requirements.txt.

## Phase 5: metrics (audit critical #4, important #8–11)
- Unknown handling:
  - keep the current metrics, labelled clearly as "excluding unknown";
  - add a second set, "unknown = error": unknown counts as a wrong prediction (a missed threat for malicious records, a false alarm for safe ones). It is the primary F1 in Compare;
  - show the unknown count next to it.
- Binary metrics per channel (email / SMS), from the channel field: precision, recall, F1, FPR, FNR, with support counts.
- Uncertainty:
  - bootstrap 95% confidence intervals (1000 resamples, fixed seed, stratified by class and channel, group-aware when group_id exists) for precision, recall, F1 and FPR, overall and per channel;
  - store them in the run JSON and display them as "0.893 [0.825; 0.946]".
- Pairwise comparison: McNemar's test (exact binomial when the discordant count is small) for any two runs on the same dataset_fingerprint, as a pure backend function. If the fingerprints differ, return "not comparable".
- Timing:
  - store per record: prompt_eval_count, eval_count, prompt_eval_duration, eval_duration, load_duration, total_duration;
  - compute tokens/s for prefill and for generation, separately;
  - compute throughput from the sum of inference times, not from wall time that includes UI rendering; keep wall time as a separate field;
  - exclude failed requests and timeouts from the latency statistics, and report their count and rate separately.
- (optional) Base-rate-adjusted precision at threat prevalence of 1% and 5% (from recall and FPR), and a cost-weighted error with configurable FP/FN costs (defaults FP=1, FN=10, documented as an assumption).
- (optional) If the installed Ollama version supports token logprobs, store the probability of the predicted label as a score and add ROC and PR curves (with AUC). If it does not, say so and skip. Never fake a score.

## Phase 5b: configurable inference options
- Make these Ollama options configurable per run and per batch cell:
  - num_ctx
  - num_thread
  - num_predict (output token cap)
  - num_batch
  temperature stays 0 and seed stays fixed; do not expose top_p/top_k (irrelevant at temperature 0).
- Defaults:
  - num_ctx: the current value
  - num_thread: Ollama's default, stored as the actual resolved value if Ollama reports it, otherwise null
  - num_predict: a documented cap large enough for fewshotcot. Measure the longest CoT output in existing runs and add margin.
  - num_batch: Ollama's default
- Validation:
  - num_ctx must fit the longest prompt in the sample (template + header + body); otherwise refuse to start and name the record that is too long;
  - num_thread must be ≤ logical cores.
- If an output hits num_predict, mark the record "truncated_output", count it in the run summary, and never silently parse a cut-off answer.
- The options actually sent are part of the run's identity. Compare and the optimum analysis treat runs with different options as different configurations, and show the differing options.
- Run tab: an "advanced" expander with these options, same design system.
- Tests: validation, truncated_output detection.

## Phase 6: batch mode (audit important #12)
- Experiment definition file (YAML or JSON, your choice, documented) under experiments/. It lists:
  - models: several tags of the same model are allowed, for quantization comparisons;
  - prompt variants;
  - dataset choice, sample size, seed;
  - repetitions;
  - start mode;
  - constraint profile;
  - an optional "options" block (Phase 5b). A list of values for one option (e.g. num_thread: [4, 8, 16]) expands into extra matrix cells.
- CLI: `python -m backend.batch experiments/<file>`:
  - runs the full matrix headless, one run JSON per cell, plus a batch summary JSON linking them via a batch_id;
  - before starting, prints the expanded cell count and the estimated total time, based on measured latencies of the same model + variant where available (otherwise says the estimate is rough);
  - resumable: on restart, cells that already have a complete, valid run file are skipped;
  - one failed cell is logged and the batch continues;
  - progress and ETA go to the console and a log file;
  - other models are unloaded between cells according to the start mode (Phase 4);
  - it must keep working with the laptop lid open and the app closed; document how to run it overnight on Windows (e.g. power settings) in README.
- Example experiment files for my real setup:
  - experiments/pilot.yaml: all installed models × 3 prompt variants, 250 records, warm start, laptop profile;
  - experiments/main.yaml: each model with one prompt variant (placeholder, to be filled in after the pilot), 1000 records, warm start, laptop profile;
  - experiments/latency_variance.yaml: 2 models, 100 records, 3 repetitions.
- (optional) Run tab: a "Batch" section that loads an experiment file, shows the matrix and estimated time, and starts it. If Streamlit cannot run it without blocking the UI, recommend the CLI and say why.
- After this phase: set protocol_version = 2 for all new runs, run a 50-record smoke batch (2 small models × 1 variant) to prove the full pipeline works end to end, delete those runs, then STOP and tell me I can start the pilot.

## Phase 7: optimum analysis (audit critical #3)
- Constraint profiles (from Phase 4) become editable in the UI.
- A pure backend function that, for a set of runs, returns:
  - the Pareto front over: F1 "unknown = error" ↑, model memory ↓, P95 latency ↓;
  - for each profile: the feasible runs and the best one by F1, ties broken by memory, then latency;
  - for each profile: McNemar's test (Phase 5) between the winner and every other feasible run on the same dataset_fingerprint. Runs with p ≥ 0.05 are marked "not significantly different from the winner". Do not use CI overlap as a significance test.
- Only runs with protocol_version 2, not dev, and a matching dataset_fingerprint enter the analysis; anything excluded is listed with the reason.
- Compare tab:
  - a "Recommendation" section: per profile, the recommended model + prompt + options, why, and which alternatives are not significantly different;
  - a Pareto scatter of F1 vs model memory, with Pareto points highlighted and a dashed memory-budget line per profile;
  - a "pairwise" table of McNemar p-values for the selected runs;
  - the existing scatters stay.

## Phase 8: non-LLM baseline (audit recommendation #16)
- A TF-IDF + logistic regression baseline, scikit-learn, fixed seed.
- Train on records NOT in the evaluation sample: exclude all sampled source_ids and, for seed+rephrased, all records sharing their group_id, so there is no leakage.
- Evaluate on exactly the same sample as the LLM runs.
- Store it as a normal run JSON: model name "baseline-tfidf-lr", same manifest, same metrics.
- Resources: measure prediction only, with the same profiler pointed at the Python process. Report training time and training memory separately.
- It appears in Compare and in the optimum analysis like any other run, clearly labelled as a baseline.
- It uses predicted probabilities as scores, so ROC/PR work for it.
- Document clearly, in OVERVIEW.md and in the Compare table label, that the baseline is trained in-distribution on the same datasets while the LLMs are not trained on them. The comparison is a reference point, not a fair contest.

## Phase 9: Compare / Results / Run / Dataset UI (same design system)
Compare:
- Legacy runs (protocol_version 1) and dev runs are hidden by default, behind a "show legacy and dev runs" toggle. When shown, they are visibly marked, and they can never be bold "best" or enter the recommendation.
- A run with 100% unknown, or with no valid predictions, can never be "best" in any column.
- Table columns:
  - model, quantization, prompt, differing options, start, n;
  - F1 with CI, recall, precision, FPR, unknown %;
  - p95 latency, throughput, tokens/s;
  - model memory, peak RSS, mean CPU, fits 8 GB / fits 16 GB;
  - peak VRAM and GPU offload only when any selected run has GPU data.
  Bold the best value per column (minimum for cost columns).
- Runs on different dataset fingerprints trigger a warning above the table.

Results:
- a "Detection" table with CI and both unknown modes;
- per-channel metrics with CI;
- a "Resources" table;
- a "RAM / CPU over time" line chart from the stored samples (a muted note if there are no samples);
- ROC/PR curves if scores exist;
- "Hardware" and "Manifest" expanders.

Run:
- a "sample size" select: 50, 100, 250, 500, 1000, custom (a number input appears when custom is chosen). Show the resulting per-cell counts and the estimated time next to it, and the "smoke test, not for results" label for 50 and 100;
- a "start" select (warm/cold);
- the loaded-models warning with the unload option;
- the "advanced" options expander (Phase 5b);
- the ETA.

Dataset: the preprocessing report from Phase 2.

## Phase 10: documentation (audit important #15)
- Fix every documentation mismatch from the audit (app.py:5, README.md:48/74/101/129/158, OVERVIEW.md:35, §8, §9, pie charts, KPI cards). Text must describe only what the code actually does.
- OVERVIEW.md must fully describe:
  - the data pipeline: input format, truncation, deduplication, spam exclusion and its rationale, grouping, balanced channel sampling;
  - the experiment design (pilot → main, latency variance);
  - the manifest and how to reproduce a run;
  - resource profiling, with its limitations: RSS vs model memory, mmap, shared GPU memory, background processes. Note that the reference results are CPU-only on a Ryzen 7 5700U;
  - inference options and their effect on resources;
  - all metrics with formulas: both unknown modes, per-channel metrics, bootstrap CI, McNemar (and base-rate precision if implemented);
  - batch mode;
  - the constraint profiles, the headroom factor and the optimum rule;
  - the baseline and its in-distribution caveat;
  - the JSON schema for protocol_version 2;
  - the legacy protocol, described as different.
- README.md: a short overview, setup, how to run one evaluation and a batch (incl. overnight on Windows), and links to OVERVIEW.md.
- Write the methodology parts in a form I can adapt into the thesis: precise, with formulas, no marketing language.

## Final report (after Phase 10)
- A summary table mapping each audit item #1–23 and recommendation #1–16 to: done / partly / not done / optional-skipped, with a file reference.
- The test count and coverage of backend/.
- One real end-to-end check on this machine: a small batch (2 small models × 1 variant × 50 records, warm). Show:
  - the resulting Compare recommendation;
  - one full run JSON (truncate the predictions and samples lists);
  - the batch summary.
  Delete these test runs afterwards, unless I say otherwise.
- A list of what I should decide next (e.g. the prompt variant per model for main.yaml, whether to add quantization) and what I need to run myself.
