"""Unit tests for data loader and PII sanitization."""

import pytest
from src.data.loader import sanitize_pii, extract_turn_pairs
import pandas as pd


def test_sanitize_pii_emails():
    text = "Please contact me at test.user@example.com for further assistance."
    clean = sanitize_pii(text)
    assert "[EMAIL]" in clean
    assert "test.user@example.com" not in clean


def test_sanitize_pii_phones():
    text = "Call me at +1 555-123-4567 or (555) 987-6543."
    clean = sanitize_pii(text)
    assert "[PHONE]" in clean
    assert "555-123-4567" not in clean


def test_sanitize_pii_ips():
    text = "Server IP is 192.168.1.100 on port 80."
    clean = sanitize_pii(text)
    assert "[IP_ADDRESS]" in clean
    assert "192.168.1.100" not in clean


def test_extract_turn_pairs_structure():
    mock_conv = [
        {"role": "user", "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "You can use list.reverse() or slicing [::-1]."},
        {"role": "user", "content": "No, that's not what I asked!"},
        {"role": "assistant", "content": "I apologize. Could you clarify?"},
        {"role": "user", "content": "Nevermind, I figured it out."},
        {"role": "assistant", "content": "Great, let me know if you need anything else."},
    ]
    df = pd.DataFrame([{
        "conversation_id": "test_conv_1",
        "model": "gpt-4",
        "timestamp": "2024-01-01T00:00:00",
        "language": "English",
        "turn": 3,
        "conversation": mock_conv,
        "redacted": True,
    }])

    pairs = extract_turn_pairs(df)
    # Turn 1: user_prompt="How do I...", assistant="You can use...", next_reaction="No, that's not..."
    # Turn 2: user_prompt="No, that's not...", assistant="I apologize...", next_reaction="Nevermind, I..."
    assert len(pairs) == 2
    assert pairs.iloc[0]["turn_idx"] == 1
    assert pairs.iloc[0]["next_user_reaction"] == "No, that's not what I asked!"
    assert pairs.iloc[1]["turn_idx"] == 2
    assert pairs.iloc[1]["next_user_reaction"] == "Nevermind, I figured it out."
