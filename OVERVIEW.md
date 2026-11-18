# MarkerPhish — Methodology and Technical Reference

This document describes what the code does, precisely enough to reproduce a run and to adapt the method
sections into the thesis ("Platform for evaluating language models in phishing and smishing detection with a
focus on resource efficiency"). Numbers quoted here were measured on the reference machine (section 13).

Contents

1. Task and architecture
2. Data pipeline
3. Sampling
4. Prompt and context budget
5. Inference and inference options
6. Experiment design
7. Metrics
8. Resource profiling
9. Constraint profiles and the optimum rule
10. Non-LLM baseline
11. Reproducibility manifest and how to reproduce a run
12. Batch mode
13. Reference hardware, known limitations and measurement notes
14. Adding a new model
15. Run file schema (protocol 2) and the legacy protocol
16. Module map

---

## 1. Task and architecture

Each message (an email from PhishFuzzer or an SMS from the Mendeley smishing corpus) is classified by a local
language model served by Ollama as **malicious** (phishing / smishing) or **valid**. The positive class is
malicious. The channel (email / SMS) is known from the dataset and is never predicted by the model.

```
Data/  ──> backend/data_loader + preprocessing ──> backend/sampling ──> sample (n records)
                                                                            │
           Ollama (/api/generate, temperature 0) <── backend/evaluator <────┘
                    │                                   │  ResourceSampler (backend/profiler, 0.5 s)
                    ▼                                   ▼
           raw predictions + timings + resources ──> backend/metrics, detection, stats
                                                     backend/run_report (+ manifest)
                                                                  │
                                                     results/eval_*.json
                                                                  │
           Streamlit UI (app.py, frontend/) <── Compare / optimum (backend/compare, optimum)
```

The UI (`app.py`, `frontend/`, `ui/`) only renders; all logic is in `backend/` as testable functions. The unit
tests (`tests/`, pytest) never contact a running model: an in-process fake Ollama replaces the HTTP calls.

---

## 2. Data pipeline

Preprocessing version: `config.PREPROCESSING_VERSION = "3"` (stored in every run and part of the dataset cache
key). Per source the steps are, in order:

1. **Read** the rows and assign identifiers.
   - `source_id`: `seed:<No.>`, `rephrased:<No.>` (PhishFuzzer), `sms:<row>` (1-based CSV data row, header
     excluded).
   - `group_id`: `pf:<Original_ID>` for PhishFuzzer (a seed email uses its own `No.`, so a seed email and its six
     rephrasings share one group); `sms:<row>` for SMS (every SMS is its own group).
2. **Exclude spam** (`config.PHISHFUZZER_EXCLUDED_TYPES = ("Spam",)`, `config.SMISHING_EXCLUDED_LABELS =
   ("spam",)`) and any unrecognized label (counted separately).
3. **Build the model input** (section 2.1) and drop records with neither a subject nor a body.
4. **Deduplicate** within the source (section 2.3), on the untruncated input.
5. **Truncate** the body (section 2.2); the truncated text is what the model sees.

Labels are mapped to `phishing` / `smishing` / `valid` and then to the binary classes
(phishing, smishing → malicious; valid → safe).

### 2.1 Model input format

Email (fixed format):

```
From: <Sender>
Subject: <Subject>
Links: <url 1> <url 2> ...
<empty line>
<Body>
```

- `Sender`, `Subject`: cleaned (`utils.clean_text`: HTML → text with link targets kept, entity unescaping,
  NFKD normalisation, non-printable characters removed, whitespace collapsed); `(unknown)` / `(none)` when empty.
- `Links`: the URLs of PhishFuzzer's `URL` field (a Python list stored as a string, sometimes nested), flattened
  and deduplicated in first-seen order, separated by single spaces; `none` when the field is empty.
- SMS: the cleaned text only.

### 2.2 Truncation

Header and body together are limited to `config.MESSAGE_MAX_CHARS = 3,945` characters (derivation in section 4.2).
Only the body is cut; the header is never cut. A cut body ends with ` [...]`. Per record the run stores
`truncated`, `original_length` (characters of the untruncated input) and `chars_removed`.

When a body is cut, URLs that appear **only** in the cut-off part (not in the Links header and not in the kept
part of the body) are appended to the Links header in order of appearance (`links_moved` per record). Moving
links lengthens the header and so cuts more body; the step repeats until no further link is lost. Two limits
bound it:

- `LINKS_MIN_BODY_CHARS = 1000`: the header may grow only while at least 1,000 body characters stay visible
  (or the whole body, if shorter). Without it a newsletter with 599 body links (`seed:2212`) produced a
  12,021-character header.
- `LINKS_MOVED_MAX_CHARS = 1000`: moved links take at most 1,000 characters per email. Newsletter tracking URLs
  tokenize at about 1.2 characters per token on the densest tokenizer measured; without the cap `seed:2969`
  needed 4,047 of 4,096 context tokens.

Links that do not fit stay lost and are counted (`body_urls_lost`). PhishFuzzer's `URL` field does not list every
link that appears in a body, which is why the move is needed at all.

Result (whole dataset, before sampling):

| source | raw | spam removed | exact dupes | normalized dupes | label conflicts | final | truncated | median / max chars cut | links moved (emails) | emails still losing links |
|---|---|---|---|---|---|---|---|---|---|---|
| phishfuzzer_seed | 3,300 | 1,074 | 1 | 0 | 0 | 2,225 | 211 (9.5%) | 2,206 / 28,623 | 254 (54) | 9 (765 links) |
| phishfuzzer_rephrased | 19,800 | 6,444 | 0 | 0 | 0 | 13,356 | 187 (1.4%) | 300 / 8,260 | 14 (12) | 0 |
| mendeley_sms | 10,191 | 3,397 | 778 | 259 | 0 | 5,757 | 0 | — | 0 | 0 |

(Before this pipeline, the platform cut every message at 1,500 characters and gave the model neither the sender
nor the URL field: 31% of seed and 61% of rephrased emails were cut.)

### 2.3 Deduplication

Within each source, on the full untruncated model input:

- **exact duplicate**: identical text to an earlier record;
- **normalized duplicate**: identical after lowercasing, collapsing whitespace and stripping punctuation at both
  ends;
- **label conflict**: copies of one (normalized) text carry different labels: every copy is dropped and counted.

The first record of each duplicate group is kept.

### 2.4 Spam exclusion (deliberate decision)

Spam is excluded from both datasets, a decision agreed with the thesis supervisor (`config.SPAM_EXCLUSION_RATIONALE`).
The task is threat detection: does a message try to deceive the recipient into an action that benefits an
attacker? Spam is unsolicited but not necessarily deceptive, and both datasets treat it as a third class whose
boundary to phishing and to legitimate marketing is dataset-specific. Keeping it would force a choice between
counting spam as malicious (inflating recall with non-threats) and counting it as valid (penalising models for
flagging fraud-like spam).

**Effect on interpretation:** the false-positive rate is measured on legitimate messages only. Real traffic
contains spam, and a deployed filter may flag some of it, so the reported FPR is a lower bound on the alarm rate a
user would experience.

---

## 3. Sampling

`backend/sampling.balanced_sample(df, n, seed=42)`.

**Cells.** The sample is balanced over four (channel, class) cells, in this fixed order:
phishing email, valid email, smishing SMS, valid SMS. A dataset choice with one channel uses its two cells.

**Allocation (largest remainder with equal quotas).** With k cells, each cell gets ⌊n/k⌋ records and the
n mod k extra records go to the first cells in the fixed order. Alternating threat / valid keeps the classes within
one record of each other: n = 50 → 13 / 13 / 12 / 12 (25 malicious / 25 safe); n = 250 → 63 / 63 / 62 / 62.

**Groups.** Records sharing a `group_id` are either all in the sample or all out, so rephrasings of one email
never count as independent records. A cell's target is met exactly with whole groups: groups are visited in a
seeded random order, and a group is taken only if the remaining target can still be reached exactly by the
groups after it (suffix subset-sum reachability).

**Seeds.** Each cell has its own generator seeded by (seed, cell), so a cell's records do not depend on which
other cells exist (the SMS records of `sms` and `seed+sms` are the same). For singleton groups the selection of
size n is a prefix of the cell's permutation, so a 250-record sample is a subset of the 1,000-record sample.

**Fallback.** If a cell cannot supply its share (too few unique records, or no exact combination of whole
groups), the sample falls back to the largest balanced size every cell can supply, and the run records the
requested size, the actual size and the reason. The sample is never silently unbalanced.

Records available per cell after deduplication, and the largest balanced sample:

| dataset choice | phishing email | valid email | smishing SMS | valid SMS | max balanced |
|---|---|---|---|---|---|
| seed+sms (used for the study) | 1,125 | 1,100 | 2,366 | 3,391 | 4,401 |
| seed | 1,125 | 1,100 | — | — | 2,201 |
| sms | — | — | 2,366 | 3,391 | 4,732 |
| rephrased (groups of 6) | 6,756 (1,126 groups) | 6,600 (1,100 groups) | — | — | 13,200 |
| all (groups of 7) | 7,881 (1,126 groups) | 7,700 (1,100 groups) | 2,366 | 3,391 | 9,464 |

With groups of 6 or 7, some sizes cannot be met exactly (e.g. `all` at 50 → 28 records, 1,000 → 980); the
fallback reason says so.

**Sample sizes.** 50, 100, 250, 500, 1,000 or any integer ≥ 20; default 1,000. 50 and 100 are labelled
"smoke test, not for results"; sizes below 500 are for testing and the pilot only.

---

## 4. Prompt and context budget

### 4.1 Prompt variants

Three variants (`config.PROMPT_VARIANTS`) ask the same binary question and differ in one factor at a time:

- `zeroshot`: instructions only; answer `{"label": "malicious" | "valid"}`;
- `fewshot`: instructions + six hand-written examples (three email, three SMS; not taken from the datasets);
- `fewshotcot`: the same examples with a one-sentence justification before the label
  (`{"reasoning": "...", "label": ...}`, reasoning under 30 words).

The email examples use the same From / Subject / Links format as the evaluated messages. The answer is
constrained by a JSON Schema passed as Ollama's `format` parameter; `parse_binary_label` reads the `label` field
with fallbacks for malformed output. Models with Ollama's "thinking" capability get `think: false`. The prompt
text per channel and its SHA-256 hash are stored in every run (section 11).

### 4.2 Context budget

`num_ctx` (default 4,096 tokens) holds the prompt **and** the output. The message limit is derived as

  tokens(template + message) + num_predict ≤ num_ctx · (1 − margin),  tokens(x) ≈ chars(x) / c

  ⇒ message chars ≤ c · (num_ctx · (1 − margin) − num_predict) − template chars

with num_ctx = 4,096, margin = 10%, num_predict = 256, c = 2.4 characters per token and the longest template
(fewshotcot) = 4,287 characters, giving **3,945 characters**.

The value c = 2.4 comes from a measurement on all five installed tokenizers (Ollama `prompt_eval_count`, fewshotcot
prompt): prose-heavy emails take 3.7–4.4 characters per token, the most link-dense email 2.5–3.5. With c = 3.0 that
email overflowed num_ctx on the densest tokenizer (4,046 + 256 tokens), with 2.5 it missed the 10% margin by 36
tokens; 2.4 meets it for that email.

**Worst-case margin.** Characters cannot bound tokens for random strings (tracking IDs tokenize at 1–2 characters
per token). After the link move (section 2.2), the 12 most link-dense emails were checked on the densest tokenizer:
all fit num_ctx; two miss the 10% margin (`seed:2323` 3,740 and `seed:2913` 3,751 tokens including the 256 output
tokens). The worst observed spare is **345 tokens (8.4%)**. Two safeguards cover what the estimate cannot:
the per-request check in section 5.3 and the optional per-model calibration (section 14).

---

## 5. Inference and inference options

### 5.1 Fixed settings

Every request: `temperature = 0`, `top_p = 1`, `seed = 42`, `stream = false`, the JSON Schema `format`.
`num_gpu = 0` (CPU) unless GPU is chosen as the device (Run tab, "advanced"; default CPU). CPU runs the whole
model on the processor, so results do not depend on graphics drivers; GPU offloads model layers to an NVIDIA GPU
if one is available and otherwise falls back to CPU. Runs stored with the old device labels are shown as CPU or
GPU. Results are deterministic at temperature 0 (two runs of one
model on the same 50 records gave identical answers). The Ollama URL is `http://127.0.0.1:11434`
(`MARKERPHISH_OLLAMA_URL` overrides it); see section 13.3 for why not `localhost`.

### 5.2 Configurable options

`backend/options.resolve_options`, per run (Run tab, "advanced") and per batch cell:

| option | default | validation |
|---|---|---|
| `num_ctx` | 4,096, never above the model's `context_length` (`/api/show`) | longest prompt of the sample must fit (below) |
| `num_thread` | not sent (Ollama decides) | ≤ logical cores |
| `num_predict` | 256 for every model; a per-model override is possible | ≥ 1 |
| `num_batch` | not sent (Ollama decides) | ≥ 1 |

**Output cap (provisional).** Measured once per installed model on the longest message with fewshotcot: 32–43
output tokens, all stopped on their own. 256 is about six times the longest output.

**Context check before a run.** The longest prompt of the sample must satisfy
estimated tokens + num_predict ≤ num_ctx · 0.9, with the model's calibrated characters per token if it has one,
otherwise c = 2.4. With the default estimate every prompt fits by construction (section 4.2). If a calibrated,
denser tokenizer needs more room and `num_ctx` was not set by the user, `num_ctx` is raised in steps of 512 up to
the model's context length and the raise is recorded; otherwise the run is refused before any request and the
record is named.

**What Ollama actually used.** `num_thread` and `num_batch` are read from the runner's command line
(`-t`, `-b`) after the first request: `num_batch` is always shown (512), `num_thread` only when it was set; with
Ollama's default it is stored as null.

**Run identity.** The options sent are part of a run's configuration. Compare and the optimum analysis treat runs
with different options as different configurations and show the options that differ.

**Effect on resources.** `num_ctx` sets the KV-cache size, which is part of the model memory reported by Ollama;
`num_thread` sets how many cores inference uses (CPU share, latency); `num_batch` changes prompt-processing speed
and buffer memory; `num_predict` only caps the output.

### 5.3 Per-request checks

- **Cut-off output:** `done_reason == "length"` → `truncated_output = true`; the answer is never parsed and counts
  as unknown.
- **Possibly cut input:** `prompt_eval_count + num_predict ≥ num_ctx` → `input_possibly_truncated = true` (Ollama
  shortens a prompt that does not fit). `prompt_eval_count` counts the whole prompt even when its prefix is served
  from Ollama's prompt cache (measured), so the check is reliable. The prediction is kept and the count is shown.
- **Failed request** (timeout, HTTP error, no connection): `request_status = timeout | error`; the prediction is
  unknown; the request is excluded from latency statistics.

---

## 6. Experiment design

Two-stage study on `seed+sms`, plus a latency-variance check:

1. **Pilot:** every installed model × the three prompt variants on 250 records. Purpose: choose the prompt variant
   per model and measure real latencies.
2. **Main study:** every model with its chosen variant on 1,000 records.
3. **Latency variance:** a small repeated experiment (e.g. 2 models, 100 records, 3 repetitions). Predictions are
   deterministic, so repetitions measure timing variance only.

The LLM-rephrased set (`rephrased`, `all`) is supported but not part of the main results; it can serve a separate
paraphrase-robustness experiment. Quantization (several tags of one model) is supported but not a required axis.
Runs can be made in the UI (one at a time) or as a batch (section 12).

---

## 7. Metrics

Computed from the stored per-record predictions only (`backend/detection.py`, `backend/stats.py`); a run stores
`metrics_version` and is recomputed on load when the metric code is newer, so stored runs stay usable.

### 7.1 Binary metrics and the two treatments of "unknown"

Positive class: malicious. For each record: true class y ∈ {malicious, safe}, answer p ∈ {malicious, safe, unknown}.

- **unknown = error (primary):** an unknown answer on a malicious record is a missed threat (FN), on a safe record
  a false alarm (FP).
- **excluding unknown:** unknown records are dropped from the confusion matrix (the platform's original metric).
  A model that refuses hard cases looks better under this treatment, which is why it is not primary.

  precision = TP / (TP + FP)  recall (TPR) = TP / (TP + FN)  F1 = 2·TP / (2·TP + FP + FN)

  FPR = FP / (FP + TN)  FNR = FN / (TP + FN)  specificity = TN / (TN + FP)  accuracy = (TP + TN) / (TP + TN + FP + FN)

F1 is written in the 2TP form, which equals 2PR / (P + R) wherever that is defined and is 0 (not undefined) when
TP = 0 but errors exist. An undefined rate (denominator 0) is stored as null. The unknown count and rate are
reported next to both treatments.

### 7.2 Per-channel metrics

The same metrics separately for email and SMS (from the record's channel), with support counts (malicious / safe)
and the unknown count per channel. This separates, e.g., false alarms on legitimate emails from false alarms on
legitimate SMS.

### 7.3 Bootstrap confidence intervals

95% percentile intervals for precision, recall, F1 and FPR, overall and per channel, in both treatments:

- B = 1,000 resamples, fixed seed (`stats.BOOTSTRAP_SEED`);
- stratified by (class, channel) cell: every resample keeps the original cell sizes;
- group-aware: when group ids exist, whole groups are resampled within their cell (cluster bootstrap);
- the interval is the 2.5th–97.5th percentile over resamples where the metric is defined; it is reported only
  if the metric is defined in at least half of them.

Display: `0.893 [0.825; 0.946]`.

### 7.4 McNemar's test

For two runs on the same `dataset_fingerprint` (identical records and texts), records are paired by `source_id`
and correctness uses unknown = error. With b = records only run A answered correctly and c = records only run B
answered correctly:

- b + c < 25: exact two-sided binomial test, p = min(1, 2 · P(X ≤ min(b, c))), X ~ Bin(b + c, 0.5);
- otherwise: χ² = (|b − c| − 1)² / (b + c) with 1 degree of freedom (continuity correction).

Runs on different or missing fingerprints are "not comparable". Multiple comparisons are Holm-adjusted
(section 9). Confidence-interval overlap is never used as a significance test.

### 7.5 Timing

Per record: client latency (wall time of one `/api/generate` request), `prompt_eval_count`, `eval_count`,
`prompt_eval_duration`, `eval_duration`, `load_duration`, `total_duration` (Ollama's own timings).

- Latency statistics (mean, median, P95, min, max) use successful requests only; failed requests and timeouts are
  counted separately (count and rate). In cold-start mode the first request (which loaded the model) is excluded
  and reported as `cold_start_s`; its prediction counts.
- Throughput = successful timed requests / sum of their latencies × 60 (messages per minute). Wall-clock
  throughput (including UI rendering between requests) is a separate field.
- Prefill tokens/s = Σ prompt_eval_count / Σ prompt_eval_duration; generation tokens/s = Σ eval_count /
  Σ eval_duration. Prefill tokens/s is inflated by Ollama's prompt cache: the shared template prefix is reused
  between requests (measured: a repeated prompt's prefill time dropped from 9.74 s to 0.06 s with the same
  `prompt_eval_count`). This matches a deployed filter that always uses the same prompt.

---

## 8. Resource profiling

`backend/profiler.py`.

**Process tree.** The Ollama server root is the `ollama*` process whose command line contains `serve` (not the
tray app `ollama app.exe`; fallback: `ollama*` processes whose parent is not one). Its descendants (e.g. the
`llama-server` runner that holds the model, which appears only when a model is loaded) are re-read at every sample.
Names and command lines are compared in lowercase (works on Windows, macOS, Linux).

**Sampler.** A daemon thread, every 0.5 s: process-tree RSS (sum, MB), process-tree CPU (sum of per-process CPU,
divided by the number of logical cores: 0–100% of the whole machine), NVIDIA VRAM used and GPU utilisation (NVML,
null without an NVIDIA driver). About every 2 s it reads `/api/ps` for the evaluated model (peak `size`,
`size_vram`), about every 5 s the AC power state. The time of each sampling cycle is measured (`sampler_cost_ms`;
38 ms on the reference machine, ≈ 0.5% of the machine). The sampler starts right before the first evaluation
request and stops after the last one.

**Memory metrics.**

- **Primary: model memory** = `/api/ps` size (weights + KV cache + compute buffers as allocated by the runtime).
- **Secondary: process-tree RSS.** Noisy: model weights are memory-mapped, so mapped pages count once touched and
  the OS may drop or share them (page cache); RSS also includes the server's own overhead. It can be below or above
  the model's real footprint.
- `baseline_ram_mb`: RSS right before the first evaluation request (warm start: model loaded; cold start: not yet
  loaded). `pre_warmup_ram_mb`: RSS of the server before the model load in both modes.
- GPU: `size_vram`, peak VRAM and utilisation are recorded when a GPU is used; on systems where the GPU shares
  system memory (integrated GPUs), "VRAM" is part of RAM.

**Start mode** (default warm):

- warm: one excluded warm-up request loads the model with the run's options; `cold_start_s` = its load time;
- cold: the model is unloaded (`keep_alive: 0`) and `/api/ps` is polled until it is gone (30 s timeout, recorded);
  the first request pays the load.

**Before each run** the loaded models are recorded; any model other than the evaluated one is reported
(`other_models_loaded`) and, by default, unloaded (other loaded models change both memory and latency).

**Power.** AC / battery, battery level, the Windows power plan and the Windows 11 power mode (null if unreadable)
at start and end, plus the sampler's readings. If AC power was missing at any point, `power_changed` /
`on_battery_during_run` are set and Results shows a warning (laptop CPU clocks depend on the power source).

**Background processes** are not controlled: other programs share the CPU and memory; close heavy programs for
measurement runs.

---

## 9. Constraint profiles and the optimum rule

**Profiles** (`config.CONSTRAINT_PROFILES`, editable per session in Compare):

| profile | RAM | GPU | latency limit |
|---|---|---|---|
| laptop-8gb | 8 GB | none | none (optional P95 ≤ X s) |
| laptop-16gb | 16 GB | none | none |

**Memory budget** = RAM × headroom factor, headroom = 0.5 (`MEMORY_HEADROOM_FACTOR`): 4,096 MB and 8,192 MB.
Basis: of a nominal 16 GB the reference laptop has 13.85 GB usable (the integrated GPU reserves the rest); with
ordinary desktop programs open and no model loaded 9.4 GB were in use; Windows 11 alone needs roughly 3–4 GB at
idle. The rest of the RAM is for the operating system, the application around the model, other programs and the
page cache. A run **fits** a profile when its model memory ≤ budget (and P95 ≤ the limit, if set); peak RSS is
reported alongside. A profile without a GPU must hold the whole model in RAM, so a run measured with GPU offload is
flagged.

**Optimum analysis** (`backend/optimum.analyze`):

1. Eligible runs: language-model runs with protocol 2, not dev, at least one valid prediction, on the same
   dataset fingerprint (the fingerprint shared by most eligible runs; ties: the most recent). The non-LLM baseline
   is **excluded from the recommendation** (§10). Every excluded run is listed with the reason.
2. Pareto front over F1 (unknown = error) ↑, model memory ↓, P95 latency ↓: a run is on the front when no other
   run is at least as good in all three and better in one.
3. Per profile: feasible runs; winner = highest F1, ties broken by lower model memory, then lower P95.
4. McNemar between the winner and every other feasible run; p-values Holm-adjusted over the comparisons of one
   profile (adjusted p_(i) = max_{j ≤ i} min(1, (m − j + 1) · p_(j)) over the sorted p-values). Runs with adjusted
   p ≥ 0.05 are "not significantly different from the winner".

"Not significantly different" does **not** mean equivalent: it means that the paired answers on this sample cannot
separate the two configurations. The recommendation names the configuration (model + prompt + differing options),
why it won, and these alternatives.

---

## 10. Non-LLM baseline

`backend/baseline.py`, model name `baseline-tfidf-lr`: TF-IDF (word 1–2-grams, min_df 2, at most 100,000
features, sublinear tf) + logistic regression (C = 4, liblinear, balanced class weights, random_state = sample
seed), decision threshold 0.5.

- **No leakage:** it trains on every record of the dataset choice that is not in the evaluation sample, and also
  excludes every record that shares a `group_id` with a sampled record (rephrasings of a sampled email).
- It predicts exactly the same sample, one record at a time, and is saved as a normal run (same manifest, same
  metrics). The probability of malicious is stored per record as `score`, so Results shows ROC and
  precision-recall curves for it.
- Resources: only prediction is profiled, by the same sampler pointed at the Python process; training time and
  training memory (tracemalloc peak, RSS delta) are reported separately. Its "model memory" for the optimum
  analysis is the size of the pickled vectorizer + classifier (2.7 MB on `seed+sms`).
- Run from the Run tab ("non-LLM baseline", seconds, never contacts Ollama) or `python -m backend.baseline`.

**Caveat:** the baseline is trained **in-distribution**, on the same datasets the sample is drawn from, while the
language models were never trained on them. Its numbers are a reference point, not a fair contest (on the
1,000-record `seed+sms` sample: F1 0.974, about 1.4 ms per message). Compare labels it
"baseline · trained in-distribution". It is therefore **excluded from the recommendation**: it never enters the
Pareto front or a profile's ranking and is listed among the excluded runs ("non-LLM baseline").

---

## 11. Reproducibility manifest and how to reproduce a run

Every run file has a `manifest` block:

- `protocol_version`, `dev`, `preprocessing_version`;
- `dataset`: dataset choice, requested and actual size, fallback and reason, sample seed, group-aware flag, number
  of groups, allocation order, per-cell counts, `dataset_fingerprint`, and the list of sampled `id` /
  `source_id` / `group_id`;
- `prompt`: variant, the exact template per channel (with `{{MESSAGE}}`), the output JSON Schema and
  `prompt_template_hash` (SHA-256 of the canonical JSON of both);
- `model`: name, tag, digest, size, family, parameter size and count, quantization, format, context length;
- `inference`: the options exactly as sent, how each was chosen (default / user / override / capped /
  auto-raised), the characters per token used, the longest prompt, `num_thread` / `num_batch` Ollama used,
  the number of requests whose options changed (GPU → CPU fallback);
- `software`: Ollama version and URL, git commit + dirty flag + changed files (saved runs under `results/` do not
  count; null without git), Python version, OS;
- `power`: at start, at end, `power_changed`, `on_battery_during_run`;
- `timestamps`: start and end, ISO 8601 with timezone.

`dataset_fingerprint` = SHA-256 over the sorted (source_id, text, label) of the sampled records: two runs with the
same fingerprint saw exactly the same inputs.

**To reproduce a run:** check out `software.code.commit`, install `requirements.txt`, make sure
`ollama show <model>` reports the stored digest, select the same dataset choice and sample size (the seed is
fixed), the same prompt variant, start mode and options, then run. The Results tab's "manifest" expander lists any
difference between the run and the current code commit, model digest, preprocessing version or prompt hash.

---

## 12. Batch mode

`python -m backend.batch <experiment.yaml> [--dry-run]` runs a matrix of runs without the UI.

```yaml
name: example
models: installed                 # every model Ollama lists, or a list of names (several tags = quantizations)
prompt_variants: all              # or a list: [zeroshot, fewshot, fewshotcot]
dataset: seed+sms
sample_size: 250
seed: 42
repetitions: 1
start_mode: warm                  # warm | cold
unload_other_models: true
constraint_profiles: [laptop-8gb, laptop-16gb]
options:                          # optional; a list expands the matrix
  num_thread: [4, 8]
```

- Cells = models × prompt variants × combinations of list-valued options × repetitions.
- Before starting it prints the number of cells, how many are already complete, and the estimated time (mean
  latency of earlier non-inflated protocol-2 runs of the same model + variant; otherwise labelled rough), and warns
  when on battery. `--dry-run` stops there and sends no request.
- Resumable: the batch id is a hash of the definition (with the resolved model list), the cell id a hash of
  everything that defines one run; a saved run with the same ids and the full number of records is skipped.
- One failed cell is logged and the batch continues; unknown model names stop it before it starts.
- Run files land in `results/` (the same files the UI reads), each with a `batch` block; the summary
  (`results/batches/batch_<id>.json`, rewritten after every cell, with power before and after each cell) and the
  log are in `results/batches/`.
- Running overnight on Windows: see README.

---

## 13. Reference hardware, known limitations and measurement notes

### 13.1 Reference hardware

All thesis results: AMD Ryzen 7 5700U (8 cores / 16 threads), integrated AMD GPU, no NVIDIA GPU, 16 GB RAM of which
13.85 GB usable, Windows 11, Python 3.13.3, Ollama 0.40.13, **CPU-only inference** (`num_gpu = 0`). GPU and VRAM
code paths work when an NVIDIA GPU exists, but no result depends on them.

### 13.2 Limitations

- RSS vs model memory, memory mapping and the page cache: section 8.
- Shared GPU memory: on integrated GPUs, VRAM is part of RAM.
- Background processes and the OS scheduler add noise to latency and CPU share; predictions are unaffected.
- Prefill tokens/s includes prompt-cache reuse (section 7.5).
- The output cap of 256 tokens and c = 2.4 are provisional, measured on the models installed in November 2026.
- The dataset is English and partly LLM-generated (rephrased set); public datasets may be in the models' training
  data (contamination cannot be ruled out).

### 13.3 Runs with inflated latency

Until 2026-11-14 the platform called Ollama at `http://localhost:11434`. On Windows, `localhost` resolves to IPv6
`::1` first while Ollama listens on IPv4 only, and every request waited about 2 s for the fallback (measured:
2,055–2,074 ms per call vs 4–28 ms to `127.0.0.1`). Every latency and throughput measured before the fix — all
legacy runs and any run without `manifest.software.ollama_url` — is inflated by about 2 s per message; detection
metrics are unaffected. Such runs carry `latency_inflated = true` when loaded, Results shows a warning and Compare
marks their latency columns with `*`. After the fix, the same 20-record run took 51 s instead of 95 s (mean
latency 2.55 s instead of about 4.7 s), with identical predictions.

---

## 14. Adding a new model

1. `ollama pull <model>`. Nothing in the code, the configuration or the UI lists model names: the Run tab, the
   batch (`models: installed`) and Compare take models from Ollama's `/api/tags`, metadata from `/api/show`.
   Missing metadata (quantization, context length, …) is stored as null and never stops a run.
2. Run it. `num_ctx` never exceeds the model's own context length; if the longest prompt cannot fit, the Run tab
   names the record before anything is sent. The per-request check flags any record whose input may have been cut.
3. Optional calibration: `python -m backend.calibrate <model>` sends at most three prefill-only requests (the
   longest email, the email with the longest Links header, the longest SMS), stores the densest measured
   characters per token × 0.95 in `model_overrides.json`, and unloads the model. Without calibration the Run tab
   shows a note and uses c = 2.4; it never blocks a run.
4. Optional per-model output cap: `"num_predict": <n>` for the model in `model_overrides.json`. The value used is
   stored in every run's manifest.
5. Several tags of one model (e.g. Q4 and Q8) are separate models: list both in a batch to compare quantizations;
   Compare shows the quantization column.

---

## 15. Run file schema (protocol 2) and the legacy protocol

A run file `results/eval_<model>_<variant>_<YYYYMMDD_HHMMSS>.json` (names never overwrite each other):

| key | content |
|---|---|
| `model_name`, `prompt_variant`, `execution_mode` | identification |
| `manifest` | section 11 |
| `metrics_version` | version of the detection / timing computation |
| `detection` | `overall` and `per_channel[email|sms]`, each in `unknown_as_error` and `excluding_unknown`: TP, FN, FP, TN, unknown, precision, recall, F1, FPR, FNR, specificity, accuracy, support, `ci` (precision, recall, F1, FPR: low, high, valid resamples); `unknown_predictions`, `unknown_rate`, `bootstrap` settings |
| `timing` | requests, successful / failed / timeouts / errors, failure rate, cold requests excluded, truncated outputs, possibly cut inputs, `latency_s` (mean, median, p95, min, max), sum of latencies, throughput, wall time, wall throughput, mean prompt / output tokens, prefill / generation tokens/s, server total duration |
| `resources` | peak / mean / baseline / pre-warm-up RAM (MB), peak / mean CPU (% of machine), model memory and VRAM share (MB, `/api/ps`), model file size (MB), peak / mean VRAM (MB), mean GPU utilisation (%), sample interval (s), number of samples, `/api/ps` polls, sampler cost (ms), start mode, cold start (s), loaded models before the run and unloads, `fits_profile` per profile, `samples` [{t_s, ram_mb, cpu_pct, vram_mb}] |
| `hardware` (= `hardware_profile`) | CPU model, physical / logical cores, total and free RAM (GB) at run start, GPU name / VRAM / driver if any, OS, Python and Ollama version, scan time |
| `raw_predictions` | per record: id, source_id, group_id, channel, true / predicted label (channel-level and binary), correctness, raw response, latency (s), cold-request flag, load duration, request status, truncated output, possibly cut input, prompt / output token counts, prefill / generation / total durations (s), score (baseline) |
| `classification_metrics` | the original metrics (excluding unknown) incl. the channel-level three-class breakdown |
| `resource_metrics`, `inference_config`, `data_distribution` | kept for compatibility |
| `baseline` | baseline runs only: split, training time / memory, caveat |
| `batch` | batch runs only: batch id, cell id, cell index, repetition, experiment, constraint profiles |

**Legacy protocol (protocol 1).** The six runs from October 2026 were made under a different protocol: a
three-class prompt (phishing / smishing / valid chosen by the model), `num_ctx` 2,048, no warm-up, an unstratified
sample with another class split (e.g. 49 / 35 / 16), messages cut at 1,500 characters without sender or URL field,
and latencies inflated by the `localhost` delay. They have no manifest; `protocol_version = 1` is inferred on load,
their binary metrics are recomputed from the stored predictions (a record without a ground-truth label is excluded
and counted, never assumed valid), and they are hidden in Compare by default, never marked best and never
recommended. Runs made while the refactor was in progress were tagged `dev` and are treated the same way.

---

## 16. Module map

| module | role |
|---|---|
| `config.py` | paths, labels, spam exclusion, prompt text and variants, context budget, options defaults, profiles, protocol |
| `backend/data_loader.py`, `preprocessing.py` | per-source pipeline, input format, truncation, deduplication, dataset report |
| `backend/sampling.py` | balanced, group-aware sampling |
| `backend/evaluator.py` | Ollama client, warm / cold start, pre-run check, per-request checks, run loop |
| `backend/options.py`, `model_overrides.py`, `calibrate.py` | inference options, overrides, optional calibration |
| `backend/profiler.py`, `profiles.py`, `hardware.py` | resource sampler, constraint profiles, hardware block |
| `backend/metrics.py`, `detection.py`, `stats.py` | metrics, CIs, McNemar, Holm, timing, score curves |
| `backend/manifest.py`, `run_report.py`, `utils.py` | manifest, run file assembly, persistence, compatibility |
| `backend/compare.py`, `optimum.py` | comparison table rules, optimum analysis |
| `backend/experiment.py`, `batch.py`, `baseline.py` | batch mode, non-LLM baseline |
| `app.py`, `frontend/`, `ui/` | Streamlit UI (Dataset, Run, Results, Compare) and design system |
| `tests/` | pytest suite with a fake Ollama |
