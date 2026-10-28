"""
MarkerPhish — Configuration Constants
======================================
Central configuration for paths, models, labels, and prompt templates.
"""

import os

# ──────────────────────────────────────────────
# Project root
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ──────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────
PHISHFUZZER_DIR = os.path.join(
    DATA_DIR, "Phising", "PhishFuzzer-main", "PhishFuzzer-main"
)
PHISHFUZZER_SEED = os.path.join(
    PHISHFUZZER_DIR, "PhishFuzzer_emails_original_seed_v1.json"
)
PHISHFUZZER_REPHRASED = os.path.join(
    PHISHFUZZER_DIR, "PhishFuzzer_emails_entity_rephrased_v1.json"
)
SMISHING_CSV = os.path.join(DATA_DIR, "Smishing", "Dataset_10191.csv")

# ──────────────────────────────────────────────
# Label mapping & filtering
# ──────────────────────────────────────────────
# PhishFuzzer uses: "Phishing", "Spam", "Valid"
# Mendeley SMS uses: "smishing", "spam", "ham"
# We unify everything to lowercase: phishing, smishing, valid
PHISHFUZZER_LABEL_MAP = {
    "Phishing": "phishing",
    "Valid": "valid",
    # "Spam" is intentionally excluded
}

SMISHING_LABEL_MAP = {
    "smishing": "smishing",
    "ham": "valid",
    # "spam" is intentionally excluded
}

VALID_LABELS = {"phishing", "smishing", "valid"}

# Binary threat evaluation schema (Task 1)
# Threats: phishing (email) and smishing (SMS)
# Safe: valid legitimate communications
MALICIOUS_LABELS = {"phishing", "smishing"}
SAFE_LABELS = {"valid"}
BINARY_LABELS = ["malicious", "safe"]

BINARY_LABEL_MAP = {
    "phishing": "malicious",
    "smishing": "malicious",
    "valid": "safe",
}


def to_binary_label(label: str | None) -> str:
    """
    Map a fine-grained label to binary threat domain:
    'phishing' or 'smishing' -> 'malicious'
    'valid'                 -> 'safe'
    otherwise               -> 'unknown'
    """
    if not label:
        return "unknown"
    normalized = label.strip().lower()
    if normalized in MALICIOUS_LABELS:
        return "malicious"
    if normalized in SAFE_LABELS:
        return "safe"
    return "unknown"


def is_binary_match(true_label: str, predicted_label: str) -> bool:
    """
    Determine whether prediction correctly matches true label in binary threat domain.
    A detection is correct if:
    - true is malicious (phishing or smishing) and predicted is malicious (phishing or smishing)
    - true is valid and predicted is valid
    """
    b_true = to_binary_label(true_label)
    b_pred = to_binary_label(predicted_label)
    if b_true == "unknown" or b_pred == "unknown":
        return False
    return b_true == b_pred


# ──────────────────────────────────────────────
# Ollama configuration
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_SHOW_ENDPOINT = f"{OLLAMA_BASE_URL}/api/show"

# Default models expected to be installed
DEFAULT_MODELS = ["qwen2.5:1.5b", "llama3.2", "phi3.5"]

# ──────────────────────────────────────────────
# Inference settings (optimized for GTX 1060 / CPU fallback)
# ──────────────────────────────────────────────
# Note: On older NVIDIA GPUs like GTX 1060 (Pascal, Compute 6.1),
# Ollama's CUDA toolchain can trigger a PTX buffer overrun.
# Setting num_gpu = 0 forces CPU execution which is 100% stable and fast for 1.5B-3.8B models.
DEFAULT_NUM_GPU = 0  # 0 = CPU (Safe & Stable), None = Auto/GPU
OLLAMA_REQUEST_TIMEOUT = 120  # seconds per request
OLLAMA_NUM_CTX = 2048  # context window
OLLAMA_TEMPERATURE = 0.0  # deterministic output for evaluation
OLLAMA_TOP_P = 1.0
OLLAMA_NUM_PREDICT = 25  # short label response

# ──────────────────────────────────────────────
# Sample-size presets
# ──────────────────────────────────────────────
SAMPLE_SIZES = {
    "Tiny (50)": 50,
    "Small (100)": 100,
    "Medium (250)": 250,
    "Large (500)": 500,
    "Full dataset": None,  # None => use all records
}

# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
CLASSIFICATION_PROMPT = """You are a cybersecurity expert specializing in detecting phishing emails and smishing (SMS phishing) attacks.

Your task: classify the following message into EXACTLY ONE of three categories:
- **phishing** — a deceptive email attempting to steal credentials, personal data, or money
- **smishing** — a deceptive SMS/text message attempting to steal credentials, personal data, or money
- **valid** — a legitimate, non-malicious message

RULES:
1. Respond with ONLY the label: phishing, smishing, or valid
2. Do NOT add any explanation, punctuation, or extra text
3. Base your decision solely on the message content provided

MESSAGE:
\"\"\"
{message_text}
\"\"\"

CLASSIFICATION:"""
