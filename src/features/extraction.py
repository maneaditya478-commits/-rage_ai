"""Feature extraction module for TokenRage.

Computes linguistic, structural, repetitive, and contextual features from
assistant replies and user prompts.
"""

import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import tiktoken

from src.utils.config import CONFIG

# Initialize tiktoken encoder
ENCODER = tiktoken.get_encoding(CONFIG.tiktoken_encoding)

# Regex patterns for apologies and hedging
APOLOGY_PATTERNS = [
    re.compile(r"\b(?:i\s+)?apologize\b", re.IGNORECASE),
    re.compile(r"\b(?:i(?:'m|\s+am)\s+)?sorry\b", re.IGNORECASE),
    re.compile(r"\bmy\s+apologies\b", re.IGNORECASE),
    re.compile(r"\bas\s+an\s+ai\b", re.IGNORECASE),
    re.compile(r"\bas\s+a\s+large\s+language\s+model\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:cannot|can't|am\s+unable\s+to)\b", re.IGNORECASE),
    re.compile(r"\bi\s+do\s+not\s+have\s+the\s+ability\b", re.IGNORECASE),
]

# Regex patterns for user format / length requests
FORMAT_PATTERNS = {
    "short_or_brief": re.compile(r"\b(?:brief|short|concise|summarize\s+briefly|in\s+short)\b", re.IGNORECASE),
    "one_sentence": re.compile(r"\b(?:in\s+one\s+sentence|in\s+1\s+sentence|single\s+sentence)\b", re.IGNORECASE),
    "bullet_points": re.compile(r"\b(?:bullet\s+points|bullets|bulleted\s+list)\b", re.IGNORECASE),
    "code_only": re.compile(r"\b(?:code\s+only|only\s+code|no\s+explanation|just\s+the\s+code)\b", re.IGNORECASE),
    "json_format": re.compile(r"\b(?:in\s+json|json\s+format|output\s+json)\b", re.IGNORECASE),
}

# Regex for sentence splitting
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+(?:\s+|$)")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base)."""
    if not text:
        return 0
    return len(ENCODER.encode(text))


def count_words_and_sentences(text: str) -> Tuple[int, int]:
    """Return (word_count, sentence_count) for a given text."""
    if not text:
        return 0, 0
    words = text.split()
    sentences = [s for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
    return len(words), max(len(sentences), 1 if words else 0)


def compute_ngram_repetition(text: str, n: int = 3) -> float:
    """Compute n-gram repetition score within a text.

    Score = 1.0 - (unique_ngrams / total_ngrams).
    A score near 0 means high diversity; near 1 means high repetition.
    """
    if not text:
        return 0.0
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return float(1.0 - (unique_ngrams / total_ngrams))


def count_apologies_and_hedging(text: str) -> int:
    """Count total occurrences of apology and hedging phrases in text."""
    if not text:
        return 0
    count = 0
    for pattern in APOLOGY_PATTERNS:
        matches = pattern.findall(text)
        count += len(matches)
    return count


def evaluate_format_compliance(user_prompt: str, assistant_reply: str) -> Dict[str, Any]:
    """Check if the user requested a specific format/length, and whether the assistant followed it.

    Returns dict with:
      - format_requested: bool
      - format_type: str ('none', 'short_or_brief', 'one_sentence', 'bullet_points', 'code_only', 'json_format')
      - format_violated: bool
    """
    if not user_prompt or not assistant_reply:
        return {"format_requested": False, "format_type": "none", "format_violated": False}

    _, sentences = count_words_and_sentences(assistant_reply)
    tokens = count_tokens(assistant_reply)

    for fmt_name, pattern in FORMAT_PATTERNS.items():
        if pattern.search(user_prompt):
            violated = False
            if fmt_name == "one_sentence":
                # Violated if reply is > 2 sentences or > 80 tokens
                violated = sentences > 2 or tokens > 80
            elif fmt_name == "short_or_brief":
                # Violated if reply is excessively long (> 250 tokens)
                violated = tokens > 250
            elif fmt_name == "bullet_points":
                # Violated if reply does not contain bullet characters (- , * , 1.)
                violated = not bool(re.search(r"(?:^|\n)\s*[-*•\d+\.]\s+", assistant_reply))
            elif fmt_name == "code_only":
                # Violated if reply contains substantial prose without code blocks
                violated = "```" not in assistant_reply and len(assistant_reply.split()) > 40
            elif fmt_name == "json_format":
                # Violated if reply doesn't look like JSON
                violated = "{" not in assistant_reply and "[" not in assistant_reply

            return {
                "format_requested": True,
                "format_type": fmt_name,
                "format_violated": violated,
            }

    return {"format_requested": False, "format_type": "none", "format_violated": False}


def infer_task_type(user_prompt: str) -> str:
    """Heuristic classification of task type for confounder control.

    Categories: 'code', 'creative_writing', 'technical_qa', 'chat'
    """
    prompt_lower = user_prompt.lower()
    if any(k in prompt_lower for k in ["def ", "function", "class ", "import ", "sql", "bug", "python", "javascript", "const ", "```"]):
        return "code"
    if any(k in prompt_lower for k in ["write a story", "poem", "essay", "song", "dialogue", "creative", "novel"]):
        return "creative_writing"
    if any(k in prompt_lower for k in ["how does", "what is", "why", "explain", "calculate", "solve", "difference between"]):
        return "technical_qa"
    return "chat"


def extract_features_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract full feature set across all transition pairs in the dataset.

    Computes:
      - Token, word, and sentence counts for assistant reply and user prompt
      - Reply / Prompt ratios
      - N-gram repetition (3-gram self-overlap)
      - Apology / hedging counts
      - Format compliance signals
      - Cumulative AI tokens per conversation
      - Task types and model covariates
      - Length flags (e.g., is_long_pasted_content)
    """
    records = []

    print(f"Extracting features for {len(df):,} turn transitions...")

    # Group by conversation to compute cumulative token counts accurately
    df = df.sort_values(by=["conversation_id", "turn_idx"]).reset_index(drop=True)

    # Pre-compute cumulative assistant tokens
    current_conv = None
    cum_tokens = 0
    cum_tokens_list = []

    # Fast token caching
    assistant_tokens_list = []
    user_tokens_list = []
    
    for _, row in df.iterrows():
        c_id = row["conversation_id"]
        a_text = row["assistant_reply"]
        u_text = row["user_prompt"]

        a_tok = count_tokens(a_text)
        u_tok = count_tokens(u_text)

        assistant_tokens_list.append(a_tok)
        user_tokens_list.append(u_tok)

        if c_id != current_conv:
            current_conv = c_id
            cum_tokens = a_tok
        else:
            cum_tokens += a_tok
        cum_tokens_list.append(cum_tokens)

    df["tokens_assistant"] = assistant_tokens_list
    df["tokens_user"] = user_tokens_list
    df["cumulative_ai_tokens"] = cum_tokens_list

    for idx, row in df.iterrows():
        a_text = row["assistant_reply"]
        u_text = row["user_prompt"]
        u_next = row["next_user_reaction"]
        a_tok = row["tokens_assistant"]
        u_tok = row["tokens_user"]

        a_words, a_sents = count_words_and_sentences(a_text)
        u_words, u_sents = count_words_and_sentences(u_text)

        token_ratio = float(a_tok / max(u_tok, 1))
        word_ratio = float(a_words / max(u_words, 1))

        repetition_3gram = compute_ngram_repetition(a_text, n=3)
        apology_count = count_apologies_and_hedging(a_text)

        format_info = evaluate_format_compliance(u_text, a_text)
        task_type = infer_task_type(u_text)

        is_gpt4 = 1 if "gpt-4" in str(row.get("model", "")).lower() else 0
        is_long_pasted = len(u_next) > 2000

        records.append({
            "words_assistant": a_words,
            "sentences_assistant": a_sents,
            "words_user": u_words,
            "sentences_user": u_sents,
            "token_ratio": token_ratio,
            "word_ratio": word_ratio,
            "repetition_3gram": repetition_3gram,
            "apology_count": apology_count,
            "has_apology": int(apology_count > 0),
            "format_requested": int(format_info["format_requested"]),
            "format_type": format_info["format_type"],
            "format_violated": int(format_info["format_violated"]),
            "task_type": task_type,
            "is_gpt4": is_gpt4,
            "is_long_pasted_content": int(is_long_pasted),
        })

    feat_df = pd.DataFrame(records)
    combined_df = pd.concat([df, feat_df], axis=1)
    print(f"Feature extraction complete! Resulting shape: {combined_df.shape}")
    return combined_df
