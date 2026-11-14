# Refactor progress

Source of truth: `docs/AUDIT.md` (audit, Polish) and `docs/REFACTOR_TASK.md` (the task, verbatim).
If a session restarts, resume from the first unchecked phase below.

## Status

| Phase | State |
|---|---|
| 0. Plan | approved 2026-11-14 |
| 1. Bugs, hygiene, tests | done 2026-11-14 |
| 2. Data preprocessing | done 2026-11-14 (incl. addendum 2a, committed) |
| 3. Reproducibility manifest | in progress |
| 4. Resource profiling | not started |
| 5. Metrics | not started |
| 5b. Inference options | not started |
| 6. Batch mode, then STOP: pilot can start | not started |
| 7. Optimum analysis | not started |
| 8. Non-LLM baseline | not started |
| 9. UI | not started |
| 10. Documentation | not started |
| Final report | not started |

Rules per phase: run tests, AppTest every tab, update this file, report, wait for "continue". Never commit.

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
- [ ] manifest block (all fields), git commit + dirty, Ollama version, model details
- [ ] protocol_version (legacy=1 inferred, dev flag)
- [ ] Results: Manifest expander + differences vs current
- [ ] (optional) Reproduce button

### Phase 4: resource profiling
- [ ] Ollama process-tree matching
- [ ] ResourceSampler (0.5 s; RSS, CPU %, NVML; /api/ps every 2 s; sampler cost)
- [ ] warm / cold start
- [ ] baseline + pre-warm-up RAM
- [ ] loaded-models pre-run check, unload option
- [ ] resources block + samples
- [ ] hardware block (extends HardwareProfile)
- [ ] constraint profiles + headroom factor + fits_profile
- [ ] nvidia-ml-py in requirements

### Phase 5: metrics
- [ ] "excluding unknown" and "unknown = error" sets
- [ ] binary metrics per channel
- [ ] bootstrap CI (stratified, group-aware)
- [ ] McNemar (exact for small discordant counts)
- [ ] per-record timing fields, tokens/s, throughput from inference time, failed requests excluded
- [ ] (optional) base-rate precision, cost-weighted error
- [ ] (optional) logprob scores, ROC/PR

### Phase 5b: inference options
- [ ] num_ctx, num_thread, num_predict, num_batch configurable, validated
- [ ] truncated_output detection
- [ ] options in run identity
- [ ] Run tab advanced expander

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
