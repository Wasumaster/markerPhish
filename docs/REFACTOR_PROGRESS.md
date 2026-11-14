# Refactor progress

Source of truth: `docs/AUDIT.md` (audit, Polish) and `docs/REFACTOR_TASK.md` (the task, verbatim).
If a session restarts, resume from the first unchecked phase below.

## Status

| Phase | State |
|---|---|
| 0. Plan | approved 2026-11-14 |
| 1. Bugs, hygiene, tests | done 2026-11-14 |
| 2. Data preprocessing | done 2026-11-14 (incl. addendum 2a, committed) |
| 3. Reproducibility manifest | done 2026-11-14 (committed) |
| 4. Resource profiling | done 2026-11-14 (committed da6ec73) |
| 5. Metrics | done 2026-11-14 (autonomous, committed) |
| 5b. Inference options | done 2026-11-14 (autonomous, committed) |
| 6. Batch mode | in progress (autonomous; pilot NOT run, session rule 2) |
| 7. Optimum analysis | not started |
| 8. Non-LLM baseline | not started |
| 9. UI | not started |
| 10. Documentation | not started |
| Final report | not started |

Rules per phase: run tests, AppTest every tab, update this file, report, wait for "continue". Never commit.

## AUTONOMOUS SESSION RULES (from 2026-11-14 evening; override "stop after each phase")
The user is away. Work through Phases 5, 5b, 6, 7, 8, 9, 10 in order without waiting.
1. Decide alone, following REFACTOR_TASK.md and earlier answers; prefer thesis validity + reproducibility, then simplicity. Skip ALL (optional) items (incl. Reproduce button).
   Mark runs made before the 127.0.0.1 fix as latency-inflated (~+2 s/request): flag in loaded report, note in Results and Compare, paragraph in OVERVIEW.md.
2. Real Ollama: any single check <= 5 inference requests in total; NO pilot / main / 50-record batch (replace each with a <=5-request check); Phase 5b output cap: <= 5 requests (longest message, once per installed model), generous margin, documented as provisional; always leave no model loaded.
3. Commits allowed locally: one phase per commit after tests + AppTest pass, message "Phase N: <summary>". Never push / rewrite history / delete branches / touch earlier results/ files.
4. Log every decision in "Autonomous decisions" below (what, why, alternative, how to revert).
5. If a phase fails and cannot be fixed safely: commit nothing for it, write the problem here, stop.
6. Phase 8 baseline may train/predict on real data; keep it fast.
7. Final report here and in chat. The user will only use the app: pick a model in Run, choose 50/100/250/500/1000, start: this must work end to end in the UI. Do NOT prepare extra experiments, pilot files or commands. Calibration optional, never blocks a run.
8. On a usage limit: this file must say exactly where work stopped and what is next.
9. Flexibility (hard requirement): no hardcoded model names/lists in code, config, UI or tests; models from /api/tags, metadata from /api/show, missing metadata = null, never a crash. num_ctx never exceeds the model's context_length; per-request prompt_eval_count check flags possibly cut inputs; optional `python -m backend.calibrate <model>` stores a per-model chars-per-token override (no calibration -> note, never block). num_predict: one default + optional per-model override, recorded in the manifest. Everything scales to any number of models/runs (no fixed colours/rows per model). Tests use fake model names, missing metadata, a denser tokenizer. OVERVIEW.md: section "Adding a new model".

## Autonomous decisions
(what · why · alternative · how to revert)

D1. Test model names. What: all tests use fake models (acme-small:1b, acme-small:7b, zeta-chat:latest, omega-big:8b, bare-model:latest without any metadata); comments in code/config no longer name real models. Why: session rule 9. Alternative: keep real names in tests only. Revert: none needed (tests only).
D2. Prompt cache and token counts. What: measured (3 requests, smallest installed model): Ollama's prompt_eval_count reports the full prompt length even when the prefix is served from its prompt cache (1,100 tokens both times), but prompt_eval_duration drops (9.74 s -> 0.06 s for a repeated prompt; a different message with the same template prefix also reuses it). Consequence: the per-request overflow guard (5b) can use prompt_eval_count; prefill tokens/s is inflated by cache reuse and is documented that way (timing.note). Alternative: disable caching (not exposed for /api/generate) or estimate from client latency. Revert: n/a (a measurement).
D3. Primary metric set. What: report["detection"] has both unknown modes overall and per channel with CIs for precision, recall, F1, FPR; primary = unknown_as_error. Classic classification_metrics (excluding unknown) kept unchanged. Why: task Phase 5. Alternative: replace the old block. Revert: n/a.
D4. F1 formula. What: F1 = 2TP / (2TP + FP + FN), undefined (null) only when there is no positive and nothing flagged. Same value as 2PR/(P+R) wherever that is defined; gives 0 instead of undefined when TP = 0 but errors exist (as the old code did). Revert: backend/stats.rates.
D5. Bootstrap. What: 1000 resamples, seed 20261114 (stats.BOOTSTRAP_SEED, separate from the sample seed), stratified by class x channel, whole groups when group ids exist, percentile CI; a CI is reported only if the metric is defined in at least half the resamples. Alternative: BCa intervals (more complex, no gain at n >= 250). Revert: stats.py constants.
D6. Recompute on load. What: report["metrics_version"] (detection.METRICS_VERSION = 1); utils.ensure_report_compatibility recomputes detection + timing from raw_predictions when missing or older; the file is never rewritten. Why: pilot runs stay valid if a metric is fixed later. Revert: remove _refresh_derived_metrics.
D7. Failed requests. What: request_status per record (ok / timeout / error); failed requests are excluded from latency statistics and throughput, counted separately; their prediction is "unknown" and therefore counts as an error in the primary metrics. EvaluationRun.failed now counts failed requests (it used to count unknown answers). Revert: evaluator.run_evaluation.
D8. Throughput. What: throughput_msgs_per_min = successful timed requests / sum of their latencies x 60 (client latency includes HTTP); wall_throughput_msgs_per_min kept separately. resource_metrics also excludes failed and cold requests now. Revert: metrics.compute_resource_metrics.
D9. Latency-inflated runs (session rule 1). What: manifest.software.ollama_url is recorded from now on; report_latency_inflated() = no ollama_url recorded or "localhost" in it, so all legacy runs and any run made before the fix are flagged; the loaded report gets latency_inflated; Results shows a warning in Cost, Compare marks latency cells with * and a footnote. Alternative: a list of filenames (brittle). Revert: manifest.report_latency_inflated.
D10. McNemar pairing. What: compare_runs pairs records by source_id; correctness uses unknown = error; "not comparable" when a fingerprint is missing (legacy), differs, or the record sets differ. Exact binomial below 25 discordant pairs, chi-square with continuity correction otherwise. Holm helper for Phase 7. Revert: detection.compare_runs.
D12. Output cap (PROVISIONAL). What: DEFAULT_NUM_PREDICT = 256 for every model and variant (per-variant num_predict removed). Measured: fewshotcot, longest message (seed:23), one request per installed model: 32-43 output tokens, all done_reason "stop". 256 = ~6x the maximum. Since it equals the budgeted cap, MESSAGE_MAX_CHARS is unchanged (3,945) and the phi-type re-check (needed only if the cap rose above 256) was not triggered. Alternative: 128 (tighter) or per-variant caps (24/24/160 before). Revert: config.DEFAULT_NUM_PREDICT.
D13. num_ctx auto-raise. What: default num_ctx 4096, capped at the model's context_length; raised in 512 steps only when the longest prompt of the sample does not fit with the model's calibrated chars/token (never with the default estimate, which is what MESSAGE_MAX_CHARS is built on); a user-set num_ctx is never changed (the run is refused instead, naming the record). The raise is recorded in manifest.inference.options_resolution. Why: rule 9 (safe for any model, calibration never blocks). Alternative: always refuse. Revert: options.resolve_options.
D14. Per-model overrides. What: model_overrides.json in the project root (optional; absent = defaults), written by `python -m backend.calibrate <model>` (max 3 prefill requests: longest email, longest Links header, longest SMS; stores min chars/token x 0.95, digest, measurements) or by hand (num_predict). Revert: delete the file.
D15. Truncated output. What: done_reason == "length" -> truncated_output = true, the answer is not parsed, prediction = unknown (an error in the primary metrics); counted in timing.truncated_output. Revert: evaluator.classify_message.
D16. Possibly cut input. What: input_possibly_truncated = prompt_eval_count + num_predict >= num_ctx (prompt_eval_count counts cached tokens too, see D2); flagged and counted (timing.input_possibly_truncated), the prediction is kept. Why: it is still the model's answer on what it saw; the count makes it visible. Alternative: treat as unknown. Revert: evaluator.classify_message.
D17. num_thread / num_batch actually used. What: read from the runner command line (-t / -b) after the first request; Ollama shows -t only when num_thread is set, so with Ollama's default num_thread is null; num_batch is shown (512). Stored in manifest.inference.ollama_resolved.
D18. Removed dead config: OLLAMA_NUM_PREDICT, CLASSIFICATION_PROMPT (unused since Phase 1). inference_config.num_predict is now null (see manifest.inference.options).
D11. Optional items skipped: base-rate precision, cost-weighted error, logprob scores / ROC / PR (session rule 1).

## Checklist

### Phase 0 (pre-work)
- [x] `docs/AUDIT.md` did not exist; recovered verbatim from the audit session transcript.
- [x] Task saved to `docs/REFACTOR_TASK.md`.
- [x] Plan approved.

### Decisions made at plan approval (binding)
1. Pilot and main study use **seed+sms**. Rephrased / "all" stay supported but are not in the main results (possible later paraphrase-robustness experiment).
2. Cold start: the first request's latency is excluded from latency statistics and reported as cold_start_s; its prediction counts.
3. McNemar vs winner uses **Holm** correction; docs state "not significantly different" != "equivalent".
4. While the pilot runs (Phases 7-10): edits and mocked unit tests only. Nothing that talks to Ollama (no dev runs, no live AppTest, no /api calls that load/unload). Ask the user before any real Ollama check.
5. Context budget: num_ctx includes the output. Verify longest input (fewshotcot template + header + max body) + num_predict fits num_ctx with margin; set body limit and num_predict from that. Keep the per-request prompt_eval_count check.
6. Manifest records AC vs battery power and the active Windows power plan/mode (null if unreadable). Batch CLI warns before starting on battery.
8. (after 2a) Context margin accepted as is: worst case 8.4% spare on phi3.5 + Phase 5b per-request prompt_eval_count guard. Phase 10 must document LINKS_MIN_BODY_CHARS, LINKS_MOVED_MAX_CHARS, the 9 emails that still lose links, and the 8.4% worst-case margin in OVERVIEW.md.
10. (after Phase 3) Reproduce button skipped. Power state read at run start AND end (batch: also between cells); store power_at_start, power_at_end; power_changed true if AC was lost at any point; warning in Results; batch CLI logs it per cell.
9. Phase 5b: measure real fewshotcot output lengths on all 5 models; if the needed cap > 256, raise OLLAMA_NUM_PREDICT_CAP, let MESSAGE_MAX_CHARS recalculate, re-run the budget test, report new truncation rates, AND re-run the phi3.5 check on the most link-dense emails and report the worst-case margin. Must be final before the pilot; flag clearly in the 5b report.
7. Accepted as proposed: pilot noise accepted (main study on idle machine); results dir env override + README note to pause OneDrive; seed+sms recommended over "all"; metrics_version + recompute-on-load for v2; num_ctx 4096 with ~4,000-char body limit (to be verified), 3 chars/token estimate; num_predict measured in a Phase 5b dev run, one cap for all variants; num_thread from runner `--threads` else null; YAML experiment files; dedup on full model input before truncation; single-channel choices balance over 2 cells; largest-remainder order phishing email, valid email, smishing SMS, valid SMS.

### Phase 1: bugs, hygiene, test infrastructure
- [x] tests/ with pytest, conftest.py, shared Ollama mock
- [x] tests for parse_binary_label, compute_classification_metrics, stratified_sample, channel_for_source, ensure_report_compatibility
- [x] Compare "select all" writes the multiselect key, then reruns
- [x] runner.py try/finally resets is_running
- [x] exact model-name match for model size
- [x] st.cache_data dataset load (dataset choice + PREPROCESSING_VERSION)
- [x] ensure_report_compatibility: missing label -> unknown_ground_truth, excluded and counted
- [x] requirements.txt pinned; matplotlib/seaborn removed; Python version in README
- [x] dead code removed after grep (list in report)

### Phase 2: data preprocessing
- [x] email input format From / Subject / Links / body
- [x] body-only truncation, config constant, truncated + original_length per record, per-dataset report
- [x] dedup exact + normalized per source, label conflicts dropped, report
- [x] source_id on every record; group_id for seed+rephrased
- [x] balanced 4-cell, group-aware sampling, largest remainder, fallback, cell capacity report
- [x] spam exclusion as explicit constant, counts per source
- [x] PREPROCESSING_VERSION
- [x] Dataset tab report

### Phase 3: manifest
- [x] manifest block (all fields), git commit + dirty, Ollama version, model details
- [x] power state (AC/battery, plan, mode) — user addition
- [x] protocol_version (legacy=1 inferred, dev flag)
- [x] Results: Manifest expander + differences vs current
- [ ] (optional) Reproduce button — not done, awaits confirmation

### Phase 4: resource profiling
- [x] Ollama process-tree matching
- [x] ResourceSampler (0.5 s; RSS, CPU %, NVML; /api/ps every 2 s; sampler cost)
- [x] warm / cold start
- [x] baseline + pre-warm-up RAM
- [x] loaded-models pre-run check, unload option
- [x] resources block + samples
- [x] hardware block (extends HardwareProfile)
- [x] constraint profiles + headroom factor + fits_profile
- [x] nvidia-ml-py in requirements

- [x] power at start / end / during sampling, power_changed, Results warning (user addition after Phase 3; batch per-cell logging in Phase 6)
- [x] Ollama URL 127.0.0.1 (found: localhost adds ~2 s per request on Windows)

### Phase 5: metrics
- [x] "excluding unknown" and "unknown = error" sets
- [x] binary metrics per channel
- [x] bootstrap CI (stratified, group-aware)
- [x] McNemar (exact for small discordant counts)
- [x] per-record timing fields, tokens/s, throughput from inference time, failed requests excluded
- [x] (optional) base-rate precision, cost-weighted error: SKIPPED (session rule 1)
- [x] (optional) logprob scores, ROC/PR: SKIPPED (session rule 1)

### Phase 5b: inference options
- [x] num_ctx, num_thread, num_predict, num_batch configurable, validated
- [x] truncated_output detection
- [x] options in run identity
- [x] Run tab advanced expander

- [x] per-model overrides + optional calibration CLI (session rule 9)

### Phase 6: batch mode
- [ ] experiment file format + experiments/pilot.yaml, main.yaml, latency_variance.yaml
- [ ] `python -m backend.batch`: estimate, resumable, failure-tolerant, log, unload between cells
- [ ] README overnight-on-Windows notes
- [ ] (optional) Batch section in Run tab
- [ ] protocol_version = 2, dev flag off
- [ ] 50-record smoke batch, then delete its runs
- [ ] STOP: tell the user the pilot can start

### Phases 7-10, final report
See `docs/REFACTOR_TASK.md`; items are added here when the phase starts.

## Phase log

(Each phase appends: done, deviations, open issues.)

### Phase 1 (2026-11-14)
Done:
- `tests/`: `conftest.py` (autouse guard that refuses all real HTTP; `fake_ollama`, `results_dir`, `legacy_results_dir`, `toy_df` fixtures), `fake_ollama.py` (routes /api/tags, /api/version, /api/ps, /api/show, /api/generate), tests for parser, metrics, sampler, channel helpers, report compatibility (incl. every stored run), evaluator, and a mocked AppTest (all tabs, offline Ollama, select all, failed run resets is_running). `pytest.ini`, `requirements-dev.txt`.
- Select all: the multiselect owns key `compare_selected_files`; the button's `on_click` callback writes that key (the only point where a widget key may be written), and Streamlit reruns after the callback. Stale filenames are dropped from the state.
- `runner.py`: the run body moved to `_execute_run`; `render_evaluation_runner` wraps it in try/except/finally. Streamlit's Stop/RerunException are BaseException, so they bypass `except Exception` but still hit `finally`.
- `evaluator.find_model_size`: exact name, untagged name = `:latest`; no prefix match.
- `app.load_dataset_cached`: `st.cache_data` keyed by (dataset choice, `config.PREPROCESSING_VERSION`). `PREPROCESSING_VERSION = "1"` added now (Phase 2 bumps it).
- `ensure_report_compatibility`: missing / unknown true label -> `unknown_ground_truth` (in memory only), excluded from metrics, counted in `classification_metrics.unknown_ground_truth`.
- `requirements.txt` pinned to the installed versions; matplotlib and seaborn removed (never imported). README: Python 3.13.
- Removed dead code (grep-verified): `config.BINARY_LABEL_MAP`, `config.DEFAULT_MODELS`, `EvaluationRun.to_dataframe`, `utils.load_latest_results`, `frontend/views/`, `frontend/styles.py` (+ its re-export in `frontend/__init__.py`).

Results: 78 tests pass; backend+config coverage 83%. Live AppTest against the real Ollama: 4 tabs, no exceptions, no model loaded.

Deviations:
- "Then rerun" for select all is done by Streamlit's automatic rerun after an `on_click` callback instead of `st.rerun()`: writing the key after the widget is created raises StreamlitAPIException.
- `git rm` was used for the deleted files, so those two deletions are staged; nothing is committed.

Open issues:
- Also unused, left for Phase 5b (they change there anyway): `config.CLASSIFICATION_PROMPT`, `config.OLLAMA_NUM_PREDICT`. OVERVIEW.md still mentions `to_dataframe` and `load_latest_results` (fixed in Phase 10).
- scipy is not pinned yet; it is added when Phase 5 imports it.

### Phase 2 (2026-11-14)
Done:
- `backend/preprocessing.py` (pure): email input format `From: / Subject: / Links: / <blank> / body` (placeholders "(unknown)", "(none)", "none"; Links = URL field flattened + deduplicated, space-separated); body-only truncation with marker " [...]"; `deduplicate` (exact, normalized, label conflicts); `truncation_stats`.
- `backend/sampling.py`: `balanced_sample` (4 cells from `config.SAMPLING_CELLS`, equal split, extra records in fixed order phishing email, valid email, smishing SMS, valid SMS; whole groups via suffix subset-sum bitsets; fallback to the largest balanced size with a reason; per-cell RNG seeded by (seed, cell)); `cell_capacity`; `sample_size_purpose`.
- `backend/data_loader.py` rewritten: per-source pipeline (spam -> unrecognized -> empty -> input -> dedup -> truncate), `source_id` ("seed:<No.>", "rephrased:<No.>", "sms:<1-based data row>"), `group_id` ("pf:<Original_ID>", seed uses its own No.; "sms:<row>"), `load_dataset_with_report`. `stratified_sample` removed (replaced).
- `utils.clean_text` no longer truncates.
- config: `PHISHFUZZER_EXCLUDED_TYPES`, `SMISHING_EXCLUDED_LABELS`, `SPAM_EXCLUSION_RATIONALE`, `OLLAMA_NUM_PREDICT_CAP=256`, `CONTEXT_MARGIN_FRACTION=0.10`, `CHARS_PER_TOKEN_ESTIMATE=2.4`, derived `PROMPT_TEMPLATE_MAX_CHARS=4287` and `MESSAGE_MAX_CHARS=3945`, `TRUNCATION_MARKER`, sample-size constants, `SAMPLING_CELLS`, `PREPROCESSING_VERSION="2"`.
- Few-shot email examples rewritten in the input format (From / Subject / Links lines added; body text unchanged).
- Sidebar: sample size 50 / 100 / 250 / 500 / 1000 / custom (>= 20), default 1000, smoke-test / pilot notes. "Full dataset" option removed.
- Dataset tab: sample cells, records by step, truncation, available per cell, spam rationale; source id / group id / truncated in the browser.
- `source_id`, `group_id` carried into `raw_predictions`.

Dataset numbers (whole dataset, before sampling):
| source | raw | spam | exact dupes | norm. dupes | conflicts | final | truncated | median / max chars cut | body links lost |
|---|---|---|---|---|---|---|---|---|---|
| phishfuzzer_seed | 3,300 | 1,074 | 1 | 0 | 0 | 2,225 | 211 (9.5%) | 2,206 / 28,623 | 57 |
| phishfuzzer_rephrased | 19,800 | 6,444 | 0 | 0 | 0 | 13,356 | 187 (1.4%) | 300 / 8,260 | 53 |
| mendeley_sms | 10,191 | 3,397 | 778 | 259 | 0 | 5,757 | 0 | - | 0 |

Cells (unique records / groups) and max balanced size:
- seed+sms: 1,125 / 1,100 / 2,366 / 3,391 (all singleton groups) -> max 4,401; 50, 250, 1000 exact.
- seed: 1,125 / 1,100 -> 2,201. sms: 2,366 / 3,391 -> 4,732.
- rephrased (groups of 6): 6,756 rec in 1,126 groups / 6,600 in 1,100 -> 13,200; 50 -> 48, 250 -> 240, 1000 -> 996.
- rephrased+sms -> 9,456; 50 -> 48, 250 -> 240, 1000 -> 984.
- all (groups of 7 = seed + 6 rephrasings): -> 9,464; 50 -> 28, 250 exact, 1000 -> 980.

Context-budget verification (real Ollama, fewshotcot, num_predict 1, prompt_eval_count; models unloaded afterwards):
- 3.0 chars/token, message <= 6,004: seed:2327 (35 URLs) on phi3.5 = 4,046 + 256 > 4,096 (overflow); gemma3 3,704 and qwen2.5 3,490 also missed the 10% margin. Prose emails 3.7-4.4 chars/token on all models.
- 2.5, message <= 4,289: phi3.5 3,466 + 256 = 3,722 (fits 4,096, misses 10% margin by 36).
- 2.4, message <= 3,945 (final): phi3.5 3,375 + 256 = 3,631 <= 3,686; gemma3 3,111 + 256. llama3.1/3.2 and qwen2.5 tokenize this text less densely (fit at 2.5 already).

Deviations:
- The limit is on header + body together (`MESSAGE_MAX_CHARS`), and the body gets the rest per record. A fixed body limit sized for the worst header (3.3k chars of links) would have cut ordinary emails far more.
- chars/token is 2.4, not the 3.0 proposed in the plan: 3.0 overflowed num_ctx on real tokenizers.
- Few-shot email examples changed (fewshot and fewshotcot; zeroshot has no examples). Instruction text unchanged. Needed so the examples match the input format.
- `stratified_sample` and its tests removed; replaced by `balanced_sample` and new tests.
- Seed dedup on the full input finds 1 duplicate (the audit's body-only count was higher; the From/Links lines make most near-copies distinct).

Open issues:
- 57 seed / 53 rephrased emails lose a link that appears only in the cut-off part of the body (PhishFuzzer's URL field does not list every body link). Option: add body URLs to the Links header (longer headers, less body). Not done; user to decide.
- `OLLAMA_NUM_PREDICT_CAP=256` is an assumption until Phase 5b measures CoT outputs; `MESSAGE_MAX_CHARS` is derived from it, so lowering the cap enlarges the budget automatically, and a test guards the fit.
- With groups of 6/7, rephrased/"all" samples fall back below the requested size (reported, never unbalanced).

### Phase 2a addendum: links from the cut-off body (2026-11-14)
Requested by the user after Phase 2: when a body is cut, add to the Links header the URLs that appear only in the cut-off part and are not already listed; untruncated emails unchanged; store per record; report totals; bump PREPROCESSING_VERSION.

Done:
- `preprocessing.build_email_input`: moves such URLs (order of appearance) into the Links header; repeats until stable (a longer header cuts more body). `urls_in_text` helper. Per record: `links_moved`, `body_urls_lost`. Report: `links_moved_total`, `records_with_links_moved`, `records_with_body_urls_lost`. Dataset tab shows the first two.
- Two limits, both config constants (not in the user's request, needed for the context budget):
  - `LINKS_MIN_BODY_CHARS = 1000`: moved links may grow the header only while 1,000 body chars (or the whole body) stay visible. Without it seed:2212 (599 body links) got a 12,021-char header.
  - `LINKS_MOVED_MAX_CHARS = 1000`: moved links take at most 1,000 chars. Newsletter tracking URLs tokenize at ~1.2 chars/token on phi3.5; without the cap seed:2969 reached 4,047 tokens (prompt + 256 output; 49 left in 4,096).
- `PREPROCESSING_VERSION = "3"`. Tests: moved links, iteration, min-body limit, char cap, unchanged short emails, real-data invariants.

Numbers (seed+sms; rephrased in brackets):
- truncated: 211 seed (9.5%) [187, 1.4%], unchanged by the addendum.
- links moved: 254 in 54 emails [14 in 12].
- still lost (limits reached): 765 links in 9 newsletter-style emails [0]. One of them (seed:2327) already has a 3,298-char header from its URL field, so nothing could move.

Context check on phi3.5 (densest tokenizer), fewshotcot, after the addendum:
- the 10 emails with the longest Links lines + seed:2913/2969: all fit num_ctx 4,096.
- inside the 10% margin (<= 3,686 incl. 256 output): all but two: seed:2323 3,740 (no moved links; dense body) and seed:2913 3,751. Worst spare: 345 tokens (8.4%).
- conclusion: a chars-per-token estimate cannot guarantee the 10% margin for every record (random strings tokenize at 1-2 chars/token). It guarantees "fits num_ctx" in every measured case. Phase 5b adds the per-request prompt_eval_count guard.

Open (for the user): accept 8.4% worst-case margin + per-request guard, or lower CHARS_PER_TOKEN_ESTIMATE further (2.3 -> message 3,603 chars, more truncation; still no guarantee).
Note: for singleton-group cells the 250-record sample is a subset of the 1,000-record sample (cells take a prefix of the same seeded permutation), so pilot records' prompt_eval_count covers a quarter of the main sample.

### Phase 3 (2026-11-14)
Done:
- `backend/manifest.py`: `build_manifest` with protocol_version (config.PROTOCOL_VERSION = 2), dev (config.DEV_BUILD = True until Phase 6), preprocessing_version; dataset block (choice, requested/actual size, fallback + reason, seed, group_aware, n_groups, allocation order, per-cell counts, dataset_fingerprint, sampled id/source_id/group_id list); prompt block (variant, template per channel with `{{MESSAGE}}` placeholder, output JSON Schema, prompt_template_hash = SHA-256 of canonical JSON of templates + schema); model (name, tag, digest, size, family, parameter_size/count, quantization, format, context_length, modified_at from /api/tags + /api/show); inference (options exactly as sent incl. seed, endpoint, stream, think, count of requests whose options differed, e.g. GPU->CPU fallback); software (Ollama version, git commit + dirty + changed files, ignoring saved runs under results/; Python; OS); power (on_ac_power, battery_percent, Windows power plan GUID + English name, Windows 11 power mode or null); timestamps (ISO 8601 with offset). All collectors return None instead of raising.
- `dataset_fingerprint`: SHA-256 over sorted (source_id, text, true_label).
- `report_protocol_version` (no manifest -> 1), `report_is_dev`, `manifest_differences`, `current_environment`, `manifest_summary`.
- `backend/run_report.build_run_report`: metrics report + manifest; used by the Run tab (Phase 6 batch will use it too).
- evaluator: `seed` (config.OLLAMA_SEED = 42) in every request; run records started_at/ended_at, power at start, options sent, request fields.
- Results tab: collapsed "manifest" expander (summary table, differences from current code commit / dirty flag / model digest / preprocessing version / prompt hash, prompt templates); legacy runs show a protocol-1 note.
- Tests: `tests/test_manifest.py` (16) + 2 AppTests. 153 tests pass; backend+config coverage 91%.
- Real check: a 20-record qwen2.5:1.5b zeroshot run through `build_run_report` produced a complete manifest (saved in the scratchpad, not results/; model unloaded). It recorded on_ac_power = false: the laptop was unplugged during the run.

Deviations:
- The prompt template is stored per channel (email and SMS prompts differ in wording), and the hash also covers the output JSON Schema, since the schema is part of what constrains the model.
- Saved run files under results/ do not set the dirty flag (results/ is tracked in git, so every new run would otherwise mark the next run dirty).
- Power state is taken once, at run start. Windows 11 power mode is null on this machine (no overlay value in the registry).

Open issues:
- (optional) Reproduce button: not built; needs confirmation.
- hardware_profile is still the session-start scan (audit #23); Phase 4 replaces it with a per-run hardware block.

### Phase 4 (2026-11-14)
Done:
- `backend/profiler.py`: process matching (`select_server_roots`: ollama* with "serve" in the command line, not the tray app; fallback: ollama* processes whose parent is not ollama*; lowercase names), `descendants` (recursive), `OllamaTree` (full scan once, then `children(recursive=True)` per sample; rescan only if the root is gone, max every 10 s), `TreeReader` (RSS sum, CPU % / logical cores, kept psutil.Process objects for continuous CPU counters), `_Nvml` (pynvml; silent None without driver), `ResourceSampler` (daemon thread, 0.5 s; /api/ps every 2 s for peak size / size_vram; AC power every 5 s; sampler_cost_ms per cycle; baseline = synchronous first sample), `ollama_ps`, `read_tree_rss_mb`.
- `backend/profiles.py`: `fits_profiles` (budget = ram_gb x 1024 x MEMORY_HEADROOM_FACTOR; primary = /api/ps model memory; peak RSS alongside; optional P95 limit; flag when measured with GPU offload on a no-GPU profile).
- config: `START_MODES`, `DEFAULT_START_MODE="warm"`, `COLD_UNLOAD_TIMEOUT_S=30`, `DEFAULT_UNLOAD_OTHER_MODELS=True`, `MEMORY_HEADROOM_FACTOR=0.5` (justified in the comment), `CONSTRAINT_PROFILES` laptop-8gb / laptop-16gb (no GPU, no latency limit), `DEFAULT_CONSTRAINT_PROFILES`; `OLLAMA_BASE_URL` = 127.0.0.1 (env override `MARKERPHISH_OLLAMA_URL`).
- evaluator: `loaded_models`, `unload_model`, `wait_until_unloaded`, `other_loaded_models`, `prepare_ollama`, `power_changed`; `run_evaluation(start_mode, unload_other_models, sampler_factory, tree_rss_reader)`: pre-run check -> unload others -> pre_warmup_ram_mb -> warm-up (warm) or unload + wait (cold) -> sampler around the timed requests -> power at end. Cold: first request flagged `is_cold_request`, `cold_start_s` = its load_duration, latency excluded (`latency_excluded_cold_requests`), prediction counted.
- hardware: `HardwareProfile` + os_version, python_version, ollama_version, scanned_at; `hardware_block()` per run (fixes audit #23). Report: `hardware` block; `hardware_profile` key kept (now the same per-run scan).
- run_report: `resources` block (all requested fields + model_disk_mb, ps_polls, sampler_cost_max_ms, max_tree_processes, cold_unload_confirmed, other_models_unloaded / _unload_failed, fits_profile, samples). manifest `power` = {power_at_start, power_at_end, power_changed, on_battery_during_run}.
- UI: Run tab start select (warm/cold), loaded-models warning + "unload other models before the run (recommended)" checkbox, power warning after a run; Results: power warning, "Resources" table (incl. fits per profile), hardware expander with versions.
- requirements: nvidia-ml-py==13.615.71.
- Tests: `tests/test_profiler.py` (31) + 2 AppTests; fake Ollama now tracks loaded models (/api/ps, keep_alive 0, load_duration). 186 tests pass (stable over 3 runs); backend+config coverage 91%.

Real checks (qwen2.5:1.5b, zeroshot, 20 records, CPU):
- Found: requests to "localhost" took ~2.05 s each (IPv6 ::1 tried first; Ollama listens on 127.0.0.1 only) vs 4-28 ms to 127.0.0.1. Every latency measured before this fix (incl. all 6 legacy runs and earlier dev runs) contains ~2 s of connection delay per message. After the fix: wall time 95 s -> 51 s, mean latency ~4.7 s -> 2.55 s, p95 8.13 -> 5.65 s, mean Ollama CPU 27.7% -> 49.0%, identical predictions (F1 0.8182).
- Found: re-resolving the tree with a full process scan cost ~2.2 s per sample (Windows command lines). After OllamaTree: 38 ms per sample (max 250 ms), 103 samples in 51 s.
- Warm with llama3.2 pre-loaded: other model detected and unloaded; pre_warmup RSS 27 MB (idle server), baseline 1,150 MB, peak RSS 1,237 MB, model memory (/api/ps) 1,119 MB, file 940 MB, CPU peak 51% (8 of 16 logical threads), fits both profiles.
- Cold: unload confirmed, first request 8.37 s excluded, cold_start_s 2.74.
- Runner command line: `llama-server.exe --model <blob> --port <n>`, no --threads -> resolved num_thread will be null (Phase 5b).
- Test JSONs in the scratchpad only; results/ untouched (6 legacy files).

Deviations:
- resources.model_size_mb is the /api/ps memory (as specified); the file size moved to resources.model_disk_mb. The old resource_metrics.model_size_mb (disk size) is unchanged for compatibility.
- baseline_ram_mb is sampled right before the first evaluation request: in warm mode the model is already loaded (~1.1 GB), in cold mode it is not (~27 MB). pre_warmup_ram_mb is the server before the model load in both modes.
- AC power is also polled by the sampler (every 5 s), so a power loss in the middle of a run is caught, not only at start/end.
- The process tree includes conhost.exe children of the runner (all descendants, as specified; a few MB).
- Ollama URL changed to 127.0.0.1 (not in the prompt; necessary, see above).

Open issues:
- Sampler overhead: 38 ms per 0.5 s on one core = ~0.5% of the machine (16 threads); reported per run as sampler_cost_ms.
- Phase 9 will extend the Results/Compare resource display (time-series chart etc.).

### Phase 5 (2026-11-14, autonomous)
Done:
- `backend/stats.py`: counts in both unknown modes, rates (precision, recall, F1, FPR, FNR, specificity, accuracy), stratified group-aware bootstrap weights, percentile CIs, McNemar (exact / chi2 with continuity correction), Holm.
- `backend/detection.py`: `compute_detection` (overall + per channel, both modes, CIs), `compute_timing` (latency over successful non-cold requests, failures, tokens and tokens/s, throughput from inference time, wall throughput), `request_status`, `compare_runs` (McNemar, same fingerprint only), METRICS_VERSION.
- evaluator: per record request_status, prompt_eval_duration_s, eval_duration_s, total_duration_s (plus prompt_eval_count, eval_count, load_duration_s); metrics: report gets metrics_version, detection, timing; raw_predictions carry the timing fields.
- utils: recompute derived blocks on load; latency_inflated flag. manifest: software.ollama_url, LATENCY_INFLATION_NOTE, report_latency_inflated.
- UI: Results "Detection" (both modes, CIs, per channel) and "Cost" (timing block, latency-inflation warning); Compare: primary "F1 (unknown = error)" with CI, recall / precision / FPR in that mode, unknown %, * on latency columns of inflated runs. ui.components.fmt_ci.
- requirements: scipy==1.18.1. Tests: tests/test_stats.py (19), compat + AppTests. 210 tests pass. Live AppTest: 4 tabs, no errors, no model loaded.
Deviations: optional items skipped (D11). EvaluationRun.failed now means failed requests (D7).
Open: Phase 9 completes the Compare table (CI columns for other metrics, legacy/dev toggle, fingerprint warning).

### Phase 5b (2026-11-14, autonomous)
Done:
- `backend/options.py`: resolve_options (num_ctx / num_thread / num_predict / num_batch; temperature, top_p, seed fixed; validation: num_thread <= logical cores, num_ctx <= model context_length, longest prompt + num_predict fits num_ctx x 0.9, else refuse naming the record; auto-raise only for calibrated tokenizers), longest_prompt, required_num_ctx, resolved_from_runner, run_options / differing_options / options_label (run identity).
- `backend/model_overrides.py` (optional JSON per model), `backend/calibrate.py` (CLI, <= 3 requests, unloads).
- evaluator: options passed to warm-up and every request; truncated_output; input_possibly_truncated; num_thread / num_batch from the runner; manifest.inference.options_resolution + ollama_resolved.
- config: DEFAULT_NUM_PREDICT = 256 (provisional, D12), MODEL_OVERRIDES_PATH; per-variant num_predict removed.
- Run tab: "advanced" expander (4 number inputs, 0 = default), live resolution: options sent + notes (calibration note) or an error that disables the Run button.
- Tests: tests/test_options.py (28: validation, denser calibrated tokenizer, model without metadata, overrides file, calibration with a fake denser tokenizer, truncated output, cut input, refusal before any request) + 3 AppTests incl. an end-to-end UI run (model select -> 50 records -> Run -> report saved with 13/13/12/12 cells). 241 tests pass.
- Real checks (<= 5 requests each, nothing left loaded): fewshotcot output length on all 5 installed models (5 requests); options + truncation on the smallest model (2 requests: num_predict 8 -> done_reason "length" -> truncated_output, unknown; num_thread 4 -> runner "-t 4"); longest prompt of the real 1000 sample = 8,223 chars, ~3,427 est. tokens + 256 <= 3,686 (fits without calibration).
Open: the output cap is provisional (D12); the user may set a per-model num_predict in model_overrides.json.

