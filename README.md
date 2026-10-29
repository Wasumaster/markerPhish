# MarkerPhish — LLM Threat Evaluation & Resource Benchmarking Platform

**Platform for evaluating language models in phishing and smishing detection with a focus on resource efficiency.**

> Engineering Thesis  
> PL --> *Platforma do oceny skuteczności modeli językowych w detekcji phishingu i smishingu, z optymalizacją wykorzystywanych zasobów.*

---

## Overview

MarkerPhish is a modular Python and Streamlit platform designed for automated, reproducible evaluation of large language models (LLMs) in detecting text-based social engineering attacks (phishing emails and smishing SMS). Operating against a local Ollama daemon, the platform systematically evaluates:

1. **Binary Threat Detection Effectiveness** — Binary classification framing (Malicious threats vs. Safe communications). A threat is considered intercepted whether classified as phishing or smishing, reflecting real-world cyber defense requirements.
2. **Computational Overhead & Resource Efficiency** — Rigorous latency profiling (mean, median, P95, min/max), inference throughput (messages/minute), model storage footprint, and execution environment stability.
3. **Multi-Model Comparative Benchmarking** — Concurrent side-by-side comparison of 6 or more benchmark runs with interactive trade-off visualizations (F1 vs. P95 latency, throughput vs. model size) and per-class channel breakdowns.
4. **Hardware Telemetry & Adaptive Execution** — Real-time host CPU, RAM, and GPU detection with selectable inference environments (CPU Safe Mode for Pascal/constrained architectures vs. GPU Hardware Acceleration).
5. **Zero Chrome, Academic Interface** — Streamlit UI with injected CSS suppressing Deploy buttons, status widgets, and hamburger menus, strictly adhering to serious, academic branding with zero emojis.

---

## System Architecture

The codebase is organized into decoupled `backend/` and `frontend/` packages connected through a clean application entry point:

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

The platform provides four specialized analytical views:

### 1. Dataset Overview (`frontend/overview.py`)
- Real-time composition KPIs (Total records, Malicious threat volume, Safe legitimate volume).
- Asymmetric layout with binary threat distribution bar charts and source provenance matrix (`source` × `true_label`).
- Interactive message explorer with class-based filtering, keyword search, and full text inspection.

### 2. Live Evaluation Runner (`frontend/runner.py`)
- Real-time inference monitoring with dynamic progress bar.
- Live streaming telemetry: processed records, running binary accuracy, mean latency, and ETA timer.
- Real-time activity card detailing recent inferences, predictions, correctness, and response times.
- Automated report compilation and archive generation upon completion.

### 3. Detailed Single-Run Analysis (`frontend/single_run.py`)
- **Primary Binary Metrics**: Binary Accuracy, Threat Precision, Threat Recall, Threat F1-Score, Safe Specificity, Threat Miss Rate (FNR), False Alarm Rate (FPR), and Failed Inferences.
- **Visual Diagnostics**: Interactive binary confusion matrix heatmap and ground truth vs. predicted distribution comparisons.
- **Channel Diagnostics**: Detailed multi-class metrics (phishing vs. smishing vs. valid) and channel confusion matrix.
- **Computational Overhead**: Mean, median, P95 latency, throughput, total execution time, and latency distribution histogram.
- **Audit Log & Traceability**: Full prediction audit table with per-record latency, prompt responses, and scikit-learn report exports.

### 4. Multi-Model Comparison (`frontend/multi_comparison.py`)
- Concurrent evaluation of **6 or more benchmark JSON reports**.
- **Summary Metrics Matrix**: Dense side-by-side comparison table including Accuracy, Threat Precision, Threat Recall, Threat F1-Score, Safe Specificity, P95 Latency, Throughput, Model Size, and Hardware Mode.
- **Executive KPI Cards**: Instant identification of best-in-class models (Highest F1, Highest Recall, Lowest P95 Latency, Peak Throughput).
- **Interactive Trade-off Analysis (Plotly)**:
  - Threat F1-Score vs. P95 Latency (with model size bubble scaling).
  - Throughput (msgs/min) vs. Model Storage Footprint (MB).
- **Per-Class Channel Breakdown**: Cross-model comparison across specific attack vectors (Phishing Email vs. Smishing SMS vs. Valid Legitimate).
- **CSV Matrix Export**: Direct export of comparison tables for academic papers and external graphing tools.

---

## Prerequisites & Installation

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

### Option A: Using Windows Launcher Script
Double-click `run.bat` or execute in terminal:
```cmd
run.bat
```

### Option B: Using Streamlit CLI Directly
```powershell
streamlit run app.py
```

Once running, access the application in your browser at `http://localhost:8501`.

---

## Hardware Execution Modes

Hardware capabilities are inspected during startup via `backend/hardware.py`:

- **CPU Safe Mode (`num_gpu=0`)**: Enforces multi-threaded CPU execution. Recommended for Pascal-architecture GPUs (e.g., GTX 1060, Compute 6.1) to avoid CUDA PTX toolchain buffer overruns, guaranteeing 100% stable execution for 1.5B–3.8B parameter models.
- **GPU Hardware Acceleration (`num_gpu=None`)**: Enables CUDA kernel offloading for maximum throughput on modern GPU architectures with sufficient VRAM.

---

## Evaluation Methodology & Metrics Reference

### Binary Threat Detection Schema

| Ground Truth | Model Prediction | Classification Outcome | Cyber Defense Impact |
|--------------|------------------|------------------------|----------------------|
| `phishing` or `smishing` | `phishing` or `smishing` | True Positive (TP) | Attack Intercepted |
| `valid` | `valid` | True Negative (TN) | Legitimate Message Passed |
| `valid` | `phishing` or `smishing` | False Positive (FP) | False Alarm / Disruption |
| `phishing` or `smishing` | `valid` | False Negative (FN) | Undetected Threat (Security Breach) |

### Formal Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Binary Accuracy** | `(TP + TN) / Total` | Overall correctness across all communications |
| **Threat Precision** | `TP / (TP + FP)` | Confidence that a flagged message is genuinely malicious |
| **Threat Recall** | `TP / (TP + FN)` | Percentage of actual threats intercepted |
| **Threat F1-Score** | `2 * (P * R) / (P + R)` | Harmonic mean balancing precision and recall |
| **Safe Specificity** | `TN / (TN + FP)` | Ability to pass legitimate traffic unhindered |
| **Threat Miss Rate (FNR)** | `FN / (TP + FN)` | Rate of undetected attacks passing into inbox |
| **P95 Latency** | 95th Percentile | Response time threshold covering 95% of queries |
| **Throughput** | `(Completed / Elapsed) * 60` | Operational throughput in messages per minute |

---

## License & Academic Citation

Developed as part of an Engineering Thesis (*Praca Inżynierska*):
- **Title**: *Platforma do oceny skuteczności modeli językowych w detekcji phishingu i smishingu, z optymalizacją wykorzystywanych zasobów*
- **Author**: Wasumaster / MarkerPhish Project
- **Purpose**: Academic research, reproducibility benchmarks, and defensive LLM evaluation.
