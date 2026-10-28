# MarkerPhish — LLM Evaluation Platform

**Platform for evaluating language models in phishing and smishing detection with a focus on resource efficiency.**

> Engineering Thesis (Praca Inzynierska)  
> *Platforma do oceny skutecznosci modeli jezykowych w detekcji phishingu i smishingu, z optymalizacja wykorzystywanych zasobow.*

---

## Overview

MarkerPhish is a modular Python platform for automated, reproducible evaluation of large language models (LLMs) in detecting text-based social engineering attacks (phishing emails and smishing SMS). Operating against a local Ollama instance, the platform provides:

- **Binary Threat Evaluation** — Evaluation logic framed around binary threat classification (Malicious vs. Safe). If an email phishing threat is predicted as smishing, or vice versa, the system correctly recognizes it as a successful threat interception.
- **Hardware Telemetry & Execution Control** — Host CPU, RAM, and GPU detection with selectable inference environments (CPU Safe Mode for Pascal/constrained architectures vs. GPU Hardware Acceleration).
- **Multi-Dataset Integration** — PhishFuzzer email dataset (23k) and Mendeley SMS dataset (10k) unified with strict spam filtering.
- **Dual-Tier Metrics** — Primary binary detection performance (Accuracy, Threat Precision, Threat Recall, Threat F1-Score, Specificity) paired with secondary distribution-channel diagnostics.
- **Computational Overhead Profiling** — Inference latency (mean, median, P95), throughput, and memory footprint.
- **Interactive Monitoring Dashboard** — Streamlit interface with live inference telemetry and audit logging.
- **Result Persistence** — Structured JSON export embedding evaluation metrics and hardware profiles for reproducible academic comparison.

---

## Architecture

```
markerPhish/
├── app.py              # Streamlit dashboard (main UI entry point)
├── config.py           # Paths, constants, binary mapping, prompt template
├── data_loader.py      # Dataset ingestion, filtering, unification
├── evaluator.py        # Ollama API client & inference runner
├── hardware.py         # Hardware detection & execution environment resolver
├── metrics.py          # Binary threat evaluation & resource metrics
├── utils.py            # Logging, text cleaning, result persistence
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── results/            # Saved evaluation reports (JSON)
└── Data/
    ├── Phising/PhishFuzzer-main/...    # Email dataset (JSON)
    └── Smishing/Dataset_10191.csv      # SMS dataset (CSV)
```

---

## Prerequisites

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.10+ |
| **Ollama** | Local daemon running (`ollama serve`) |
| **Models** | Locally pulled Ollama models (e.g. `qwen2.5:1.5b`, `llama3.2`, `phi3.5`) |
| **Compute Device** | Multi-core CPU or NVIDIA GPU accelerator (CUDA) |

### Tested Benchmark Models

| Model | Parameters | On-Disk Size | Context Window |
|-------|------------|--------------|----------------|
| `qwen2.5:1.5b` | 1.5B | ~986 MB | 2048 tokens |
| `llama3.2` | 3B | ~2.0 GB | 2048 tokens |
| `phi3.5` | 3.8B | ~2.2 GB | 2048 tokens |

---

## Quick Start

### 1. Configure Python Environment & Dependencies

```powershell
cd D:\Projects\markerPhish

# Create virtual environment (if not already created)
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# or Windows CMD: .venv\Scripts\activate.bat

# Install required packages
pip install -r requirements.txt
```

### 2. Start Local Ollama Server

```bash
ollama serve
```

### 3. Launch Evaluation Platform

```bash
streamlit run app.py
```

Access the dashboard in your browser at `http://localhost:8501`.

---

## Execution Environments

MarkerPhish inspects host hardware resources at startup via `hardware.py`:
- **CPU Safe Mode (`num_gpu=0`)**: Forces Ollama to run inference across CPU threads. Recommended for Pascal-architecture GPUs (GTX 1060 sm_61) and memory-constrained hosts to eliminate CUDA PTX toolchain buffer overruns.
- **GPU Hardware Acceleration (`num_gpu=None`)**: Offloads transformer layers to NVIDIA CUDA cores. Recommended when sufficient VRAM and modern compute capability are present.

---

## Evaluation Methodology

### Binary Threat Classification (Task 1)

In security applications, the primary concern is whether a user is protected from malicious communication. Differentiating an SMS threat from an email threat is secondary to isolating the threat itself.

| Ground Truth | Model Prediction | Classification Outcome | Evaluation Impact |
|--------------|------------------|------------------------|-------------------|
| `phishing` or `smishing` | `phishing` or `smishing` | True Positive (TP) | Correct Threat Detection |
| `valid` | `valid` | True Negative (TN) | Correct Safe Pass |
| `valid` | `phishing` or `smishing` | False Positive (FP) | False Alarm |
| `phishing` or `smishing` | `valid` | False Negative (FN) | Missed Threat (Security Risk) |

### Secondary Channel Diagnostics

While headline effectiveness metrics reflect binary protection, full multi-class breakdowns (phishing vs. smishing vs. valid) are retained under `channel_metrics` and accessible via the results interface.

---

## Metrics Reference

| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **Binary Accuracy** | (TP + TN) / Total | Global proportion of correctly classified communications |
| **Threat Precision** | TP / (TP + FP) | Ratio of flagged messages that were genuinely malicious |
| **Threat Recall** | TP / (TP + FN) | Interception rate of actual threats |
| **Threat F1-Score** | 2 * (Precision * Recall) / (Precision + Recall) | Harmonic balance between precision and recall |
| **Safe Specificity** | TN / (TN + FP) | Reliability on legitimate communication |
| **Threat Miss Rate (FNR)** | FN / (TP + FN) | Percentage of unintercepted attacks |
| **P95 Latency** | 95th percentile | Latency threshold under which 95% of queries complete |
| **Throughput** | Completed / Elapsed Time * 60 | Inferences processed per minute |

---

## License

This platform is developed as part of an engineering thesis research project. Contact the author for academic or commercial licensing.
