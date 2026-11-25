"""Unit tests for Phase 2: Repeat detection, batch task detection, topic shift, and rule signals."""

import pytest
import numpy as np
from src.features.repetition_and_similarity import (
    detect_repeat_prompt,
    detect_batch_task,
    TopicShiftDetector,
)
from src.labeling.rules import (
    detect_profanity,
    detect_aggressive_caps,
    detect_negative_cues,
    extract_rule_signals,
)
from src.features.extraction import (
    count_tokens,
    compute_ngram_repetition,
    evaluate_format_compliance,
    count_apologies_and_hedging,
)


def test_repeat_prompt_detection():
    # Exactly identical prompt
    p1 = "Write a python script to download a webpage."
    p2 = "Write a python script to download a webpage."
    assert detect_repeat_prompt(p1, p2) is True

    # Near identical prompt with tiny variation (>90% similarity)
    p3 = "write a python script to download a web page."
    assert detect_repeat_prompt(p1, p3) is True

    # Distinct prompts
    p4 = "What is the capital of France?"
    assert detect_repeat_prompt(p1, p4) is False


def test_batch_task_detection():
    # Classic batch task: same instruction template, different payload
    u1 = "Fix any punctuation or capitalization errors below: Miguel Rivera loves Mickey Mouse."
    u2 = "Fix any punctuation or capitalization errors below: Tony Stark's parrot can sing."
    assert detect_batch_task(u1, u2) is True

    # Translation batch task
    t1 = "Translate the following English sentence to German: The cat is on the roof."
    t2 = "Translate the following English sentence to German: Where can I buy a ticket?"
    assert detect_batch_task(t1, t2) is True

    # Unrelated prompts should NOT be a batch task
    assert detect_batch_task("How do I sort a dictionary?", "Write an essay about Shakespeare.") is False

    # Exact repetition (>95%) is a repeat prompt, NOT a batch task
    identical = "Explain quantum computing in simple terms."
    assert detect_batch_task(identical, identical) is False


def test_batch_task_excludes_from_repeat():
    # When a batch task occurs, extract_rule_signals must NOT flag it as an irritated repeat prompt
    u1 = "Translate the following English sentence to German: The cat is on the roof."
    u2 = "Translate the following English sentence to German: Where can I buy a ticket?"
    signals = extract_rule_signals(u1, u2)
    assert signals["is_batch_task"] is True
    assert signals["is_repeat_prompt"] is False
    assert signals["rule_dissatisfaction"] is False


def test_profanity_detection():
    assert detect_profanity("This is complete bullshit, fix it now!") is True
    assert detect_profanity("What the fuck are you talking about?") is True
    assert detect_profanity("Thank you for the quick explanation.") is False


def test_aggressive_caps_detection():
    # Shouting in ALL CAPS
    assert detect_aggressive_caps("WHY ARE YOU IGNORING WHAT I SAID?!") is True
    assert detect_aggressive_caps("STOP DOING THIS RIGHT NOW") is True

    # Normal acronyms should NOT trigger aggressive caps
    assert detect_aggressive_caps("I am calling the REST API using JSON and HTTP.") is False
    assert detect_aggressive_caps("How does the GPT model compare to standard CNN architectures?") is False


def test_negative_cues_detection():
    assert detect_negative_cues("No, that is completely wrong.") is True
    assert detect_negative_cues("Stop repeating yourself!") is True
    assert detect_negative_cues("I told you twice already.") is True
    assert detect_negative_cues("Wait what?!") is True
    assert detect_negative_cues("This looks great, thank you!") is False


def test_format_compliance_evaluation():
    # One sentence constraint: satisfied vs violated
    prompt_one_sent = "Explain photosynthesis in one sentence."
    good_reply = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water."
    bad_reply = (
        "Photosynthesis is a vital biological process. Green plants use sunlight, carbon dioxide, and water to produce glucose. "
        "Oxygen is released as a byproduct. This process sustains life on Earth across all ecosystems."
    )
    res_good = evaluate_format_compliance(prompt_one_sent, good_reply)
    res_bad = evaluate_format_compliance(prompt_one_sent, bad_reply)
    assert res_good["format_requested"] is True
    assert res_good["format_violated"] is False
    assert res_bad["format_violated"] is True


def test_apology_and_hedging_counts():
    apology_text = "I apologize for the confusion. As an AI language model, I cannot provide medical advice. I am sorry."
    cnt = count_apologies_and_hedging(apology_text)
    assert cnt >= 3


def test_token_counting_and_repetition():
    text = "Machine learning is fascinating. Machine learning is fascinating. Machine learning is fascinating."
    tok_count = count_tokens(text)
    assert tok_count > 10

    rep_score = compute_ngram_repetition(text, n=3)
    assert rep_score > 0.50  # Highly repetitive text has high repetition score


def test_topic_shift_detector():
    detector = TopicShiftDetector(threshold=0.35)
    
    # Similar prompts
    p_similar_1 = ["How do I reverse a string in Python?"]
    p_similar_2 = ["What is the python syntax to slice and reverse text?"]
    sims, shifts = detector.compute_similarity_batch(p_similar_1, p_similar_2)
    assert sims[0] > 0.50
    assert shifts[0] is False or shifts[0] == 0

    # Topic shift prompts
    p_shift_1 = ["How do I reverse a string in Python?"]
    p_shift_2 = ["What ingredients do I need to make beef Wellington?"]
    sims_shift, shifts_shift = detector.compute_similarity_batch(p_shift_1, p_shift_2)
    assert sims_shift[0] < 0.35
    assert bool(shifts_shift[0]) is True
