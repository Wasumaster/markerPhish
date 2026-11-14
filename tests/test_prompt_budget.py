"""The context budget: the longest possible prompt plus the output fits num_ctx with margin."""

import re

import pytest

from config import (
    CHARS_PER_TOKEN_ESTIMATE,
    CONTEXT_MARGIN_FRACTION,
    FEW_SHOT_EXAMPLES,
    MESSAGE_MAX_CHARS,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT_CAP,
    PROMPT_VARIANTS,
    build_classification_prompt,
)


@pytest.mark.parametrize("variant", list(PROMPT_VARIANTS))
@pytest.mark.parametrize("channel", ["email", "sms", None])
def test_longest_prompt_plus_output_fits_context(variant, channel):
    prompt = build_classification_prompt("x" * MESSAGE_MAX_CHARS, channel, variant)
    estimated_tokens = len(prompt) / CHARS_PER_TOKEN_ESTIMATE
    assert estimated_tokens + OLLAMA_NUM_PREDICT_CAP <= OLLAMA_NUM_CTX * (1 - CONTEXT_MARGIN_FRACTION)


def test_every_variant_output_budget_is_within_the_cap():
    assert max(v["num_predict"] for v in PROMPT_VARIANTS.values()) <= OLLAMA_NUM_PREDICT_CAP


def test_message_budget_is_large_enough_to_be_useful():
    assert MESSAGE_MAX_CHARS >= 3500


def test_few_shot_email_examples_use_the_model_input_format():
    header = re.compile(r"^From: .+\nSubject: .+\nLinks: .+\n\n.+", re.DOTALL)
    emails = [ex["text"] for ex in FEW_SHOT_EXAMPLES if ex["channel"] == "email"]
    assert emails and all(header.match(t) for t in emails)
