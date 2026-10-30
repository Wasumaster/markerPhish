# MarkerPhish — LLM Threat Evaluation & Resource Benchmarking Platform

**Platform for evaluating language models in phishing and smishing detection with a focus on resource efficiency.**

> Engineering Thesis  
> PL --> *Platforma do oceny skuteczności modeli językowych w detekcji phishingu i smishingu, z optymalizacją wykorzystywanych zasobów.*

---

## TL;DR

MarkerPhish tests how well small AI language models, running locally on your own computer, can recognize fraudulent emails (phishing) and fraudulent text messages (smishing). For every model it measures two things: how accurately it detects attacks, and how much time and computing power it needs to do so. The results are shown in an interactive dashboard and saved as reports, so different models can be compared side by side.

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [What the Platform Does](#what-the-platform-does)
3. [System Architecture](#system-architecture)
4. [Application Views & Workflow](#application-views--workflow)
5. [Prerequisites & Installation](#prerequisites--installation)
6. [Running the Platform](#running-the-platform)
7. [Hardware Execution Modes](#hardware-execution-modes)
8. [Evaluation Methodology & Metrics Reference](#evaluation-methodology--metrics-reference)
9. [License & Academic Citation](#license--academic-citation)

---

## Background & Motivation

### 1. The Problem: Phishing and Smishing as a Persistent Threat

> **In short:** Attackers trick people with convincing messages rather than hacking software, and traditional filters cannot keep up with endlessly reworded scams.

**What phishing is.** Phishing remains one of the most common entry points for cyberattacks. Instead of exploiting software vulnerabilities, attackers exploit people: a convincing email or text message asks the recipient to click a link, log in to a fake page, confirm a payment, or share a one-time code. The message itself is the weapon, so defending against it means understanding what the text is trying to make the reader do.

**Why SMS is harder.** Smishing (SMS phishing) has grown alongside the shift to mobile communication. Text messages are short, often arrive from unknown numbers, and typically impersonate couriers, banks, government offices, or mobile operators. Their brevity makes them harder to analyze than emails, because there is little context, no headers worth inspecting, and frequently just a single shortened link.

**Why classic filters fall short.** Traditional defenses such as keyword filters, sender blacklists, and URL reputation databases react to known patterns. They struggle when attackers rephrase the same scam in endless variations, which is now trivial to do with generative AI. This creates a need for detectors that judge the **intent and meaning** of a message rather than its exact wording.

### 2. Why Large Language Models, and Why Run Them Locally

> **In short:** Language models understand what a message is trying to achieve, and running them locally keeps private messages on your own machine.

**Why LLMs fit the task.** Large language models can recognize urgency, impersonation, requests for credentials, or suspicious calls to action even in messages they have never seen before. They can do so in a zero-shot setting, with only a well-designed prompt and no task-specific training.

**Why locally.** Sending private emails and text messages to an external cloud API raises serious privacy, compliance, and cost concerns, since messages may contain personal data, financial details, or confidential business information. Running models **locally through Ollama** keeps all data on the user's own machine, works offline, avoids per-request fees, and makes experiments fully reproducible, since the same model version can be re-run with deterministic settings (`temperature = 0.0`).

**The trade-off.** Local hardware is limited. Small models (1.5B–3.8B parameters) fit on ordinary laptops and older GPUs, but it is not obvious how much detection quality they sacrifice compared to larger ones, or whether they are fast enough to be practical.

---

## What the Platform Does

> **In short:** MarkerPhish is a Python and Streamlit application that runs local models against real phishing and smishing datasets and reports both detection quality and resource usage.

MarkerPhish is a modular platform designed for automated, reproducible evaluation of large language models (LLMs) in detecting text-based social engineering attacks. It works with a local Ollama daemon and covers five areas:

**1. Binary Threat Detection Effectiveness**  
*Does the model catch attacks?*  
Results are framed as a binary decision: malicious vs. safe. A threat counts as intercepted whether the model labels it phishing or smishing, which reflects what matters in real-world cyber defense: stopping the attack, regardless of its exact channel.

**2. Computational Overhead & Resource Efficiency**  
*What does detection cost?*  
Each run records latency (mean, median, P95, min/max), inference throughput (messages per minute), model storage footprint, and execution stability.

**3. Multi-Model Comparative Benchmarking**  
*Which model is best overall?*  
Six or more benchmark runs can be compared side by side, with interactive trade-off charts (F1 vs. P95 latency, throughput vs. model size) and per-class channel breakdowns.

**4. Hardware Telemetry & Adaptive Execution**  
*What hardware is it running on?*  
The platform detects the host CPU, RAM, and GPU in real time and lets the user choose an inference mode: CPU Safe Mode for Pascal or constrained architectures, or GPU Hardware Acceleration.

**5. Zero Chrome, Academic Interface**  
*A clean, distraction-free UI.*  
Injected CSS hides Streamlit's Deploy button, status widgets, and hamburger menu, keeping a serious, academic look with no emojis.

---

## System Architecture

> **In short:** The code is split into a `backend/` that does the work (loading data, querying models, computing metrics) and a `frontend/` that displays it, both started from a single `app.py`.

**How the pieces fit.** `config.py` holds shared settings, `backend/` processes data and talks to Ollama, `frontend/` builds each dashboard tab, and `results/` stores every finished evaluation as a JSON report.

```
markerPhish/
├── app.py                      # Main Streamlit application launcher
├── config.py                   # Global constants, paths, prompt templates, binary mappings
├── run.bat                     # Windows one-click launcher script
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & quickstart
├── OVERVIEW.md                 # Detailed technical thesis documentation
├── backend/                    # Core processing & analytics engine
│   ├── __init__.py             # Backend package initialization
│   ├── data_loader.py          # Dataset ingestion, filtering & unification
│   ├── evaluator.py            # Local Ollama REST client & inference runner
│   ├── hardware.py             # Hardware scanning & execution mode resolver
│   ├── metrics.py              # Binary threat metrics & resource profiling
│   └── utils.py                # Logging, text cleaning & result persistence
├── frontend/                   # Modular presentation layer
│   ├── __init__.py             # Frontend package exports
│   ├── styles.py               # Design system, typography & chrome suppression CSS
│   ├── sidebar.py              # Configuration sidebar (model, dataset, hardware)
│   ├── header.py               # Static academic branding header
│   ├── overview.py             # Tab 1: Dataset distribution & message browser
│   ├── runner.py               # Tab 2: Live evaluation runner & telemetry
│   ├── single_run.py           # Tab 3: Detailed single-run analysis
│   ├── multi_comparison.py     # Tab 4: Multi-model comparative benchmarking (6+ runs)
│   └── views/                  # View re-exports package
├── results/                    # Persistent evaluation reports (JSON)
└── Data/                       # Raw datasets
    ├── Phising/PhishFuzzer-main/ # Email phishing dataset (JSON)
    └── Smishing/Dataset_10191.csv# SMS smishing dataset (CSV)
```

---

## Application Views & Workflow

> **In short:** The dashboard has four tabs that follow a natural workflow: look at the data, run an evaluation, analyze one result in detail, then compare several models.

### 1. Dataset Overview (`frontend/overview.py`)

*See what data the models will be tested on.*

- Real-time composition KPIs (total records, malicious threat volume, safe legitimate volume).
- Asymmetric layout with binary threat distribution bar charts and a source provenance matrix (`source` × `true_label`).
- Interactive message explorer with class-based filtering, keyword search, and full text inspection.

### 2. Live Evaluation Runner (`frontend/runner.py`)

*Run a model on the dataset and watch it work.*

- Real-time inference monitoring with a dynamic progress bar.
- Live telemetry: processed records, running binary accuracy, mean latency, and ETA timer.
- Activity card showing recent inferences, predictions, correctness, and response times.
- Automatic report compilation and archive generation when the run completes.

### 3. Detailed Single-Run Analysis (`frontend/single_run.py`)

*Understand exactly how one model performed.*

- **Primary Binary Metrics:** Binary Accuracy, Threat Precision, Threat Recall, Threat F1-Score, Safe Specificity, Threat Miss Rate (FNR), False Alarm Rate (FPR), and Failed Inferences.
- **Visual Diagnostics:** Interactive binary confusion matrix heatmap and ground truth vs. predicted distribution comparisons.
- **Channel Diagnostics:** Detailed multi-class metrics (phishing vs. smishing vs. valid) and a channel confusion matrix.
- **Computational Overhead:** Mean, median, and P95 latency, throughput, total execution time, and a latency distribution histogram.
- **Audit Log & Traceability:** Full prediction audit table with per-record latency, prompt responses, and scikit-learn report exports.

### 4. Multi-Model Comparison (`frontend/multi_comparison.py`)

*Put several models next to each other and pick the best fit.*

- Concurrent evaluation of **6 or more benchmark JSON reports**.
- **Summary Metrics Matrix:** Side-by-side table with Accuracy, Threat Precision, Threat Recall, Threat F1-Score, Safe Specificity, P95 Latency, Throughput, Model Size, and Hardware Mode.
- **Executive KPI Cards:** Instant identification of best-in-class models (Highest F1, Highest Recall, Lowest P95 Latency, Peak Throughput).
- **Interactive Trade-off Analysis (Plotly):**
  - Threat F1-Score vs. P95 Latency (with model size bubble scaling).
  - Throughput (msgs/min) vs. Model Storage Footprint (MB).
- **Per-Class Channel Breakdown:** Cross-model comparison across attack vectors (Phishing Email vs. Smishing SMS vs. Valid Legitimate).
- **CSV Matrix Export:** Direct export of comparison tables for academic papers and external graphing tools.

---

## Prerequisites & Installation

> **In short:** You need Python 3.10+, a running Ollama server, and the three default models downloaded.

| Requirement | Specification |
|-------------|---------------|
| **Operating System** | Windows 10/11, Linux, or macOS |
| **Python** | 3.10+ |
| **Ollama** | Local daemon running (`ollama serve`) |
| **Default Models** | `qwen2.5:1.5b`, `llama3.2:latest`, `phi3.5:latest` |
| **Compute Device** | Multi-core CPU or NVIDIA CUDA GPU |

### Setup Instructions

```powershell
# 1. Clone or navigate to the repository
cd D:\Projects\markerPhish

# 2. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull required Ollama models
ollama pull qwen2.5:1.5b
ollama pull llama3.2
ollama pull phi3.5
```

---

## Running the Platform

> **In short:** Double-click `run.bat` (Windows) or run `streamlit run app.py`, then open `http://localhost:8501`.

### Option A: Windows Launcher Script

Double-click `run.bat` or execute it in a terminal:

```cmd
run.bat
```

### Option B: Streamlit CLI

```powershell
streamlit run app.py
```

Once running, open the application in your browser at `http://localhost:8501`.

---

## Hardware Execution Modes

> **In short:** Choose CPU mode for stability on older GPUs, or GPU mode for speed on modern ones.

Hardware capabilities are inspected at startup by `backend/hardware.py`.

**CPU Safe Mode (`num_gpu=0`)**  
*Stable everywhere.*  
Forces multi-threaded CPU execution. Recommended for Pascal-architecture GPUs (e.g. GTX 1060, Compute 6.1) to avoid CUDA PTX toolchain buffer overruns, guaranteeing stable execution for 1.5B–3.8B parameter models.

**GPU Hardware Acceleration (`num_gpu=None`)**  
*Fastest on modern hardware.*  
Enables CUDA kernel offloading for maximum throughput on modern GPU architectures with sufficient VRAM.

---

## Evaluation Methodology & Metrics Reference

> **In short:** Every prediction is sorted into one of four outcomes (caught attack, correctly passed message, false alarm, missed attack), and all metrics are calculated from those counts.

### Binary Threat Detection Schema

**How a single prediction is scored.** The model's answer is compared with the true label of the message:

| Ground Truth | Model Prediction | Classification Outcome | Cyber Defense Impact |
|--------------|------------------|------------------------|----------------------|
| `phishing` or `smishing` | `phishing` or `smishing` | True Positive (TP) | Attack Intercepted |
| `valid` | `valid` | True Negative (TN) | Legitimate Message Passed |
| `valid` | `phishing` or `smishing` | False Positive (FP) | False Alarm / Disruption |
| `phishing` or `smishing` | `valid` | False Negative (FN) | Undetected Threat (Security Breach) |

### Formal Metrics

**What the numbers mean.** The first column says what each metric is called, the last column says what it tells you in plain terms.

| Metric | Formula | Description |
|--------|---------|-------------|
| **Binary Accuracy** | `(TP + TN) / Total` | Overall correctness across all communications |
| **Threat Precision** | `TP / (TP + FP)` | Confidence that a flagged message is genuinely malicious |
| **Threat Recall** | `TP / (TP + FN)` | Percentage of actual threats intercepted |
| **Threat F1-Score** | `2 * (P * R) / (P + R)` | Harmonic mean balancing precision and recall |
| **Safe Specificity** | `TN / (TN + FP)` | Ability to pass legitimate traffic unhindered |
| **Threat Miss Rate (FNR)** | `FN / (TP + FN)` | Rate of undetected attacks passing into the inbox |
| **P95 Latency** | 95th percentile | Response time threshold covering 95% of queries |
| **Throughput** | `(Completed / Elapsed) * 60` | Operational throughput in messages per minute |

---

## License & Academic Citation

Developed as part of an Engineering Thesis (*Praca Inżynierska*):

- **Title:** *Platforma do oceny skuteczności modeli językowych w detekcji phishingu i smishingu, z optymalizacją wykorzystywanych zasobów*
- **Author:** Szymon Wąs / MarkerPhish Project
- **Purpose:** Academic research, reproducibility benchmarks, and defensive LLM evaluation.
