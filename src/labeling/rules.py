"""Rule-based heuristic signals and frustration detectors for U_{t+1}."""

import re
from typing import Dict, Any

from src.features.repetition_and_similarity import detect_batch_task, detect_repeat_prompt

# Core profanity and hostile language patterns
PROFANITY_PATTERN = re.compile(
    r"\b(?:fuck|fucking|fucked|shit|bullshit|damn|hell|asshole|bitch|crap|pissed|idiot|stupid)\b",
    re.IGNORECASE,
)

# Frustration, correction, and negative feedback cues
NEGATIVE_CUES_PATTERN = re.compile(
    r"(?:\b(?:wrong|incorrect|again|stop|i told you|i said|i already told|i already said|not what i asked|not what i wanted|you didn't|you failed|useless|nonsense|garbage|hallucinating|listen to me|read carefully)\b|\?{2,}|\!{2,}|\?!|!\?)",
    re.IGNORECASE,
)


def detect_profanity(text: str) -> bool:
    """Check if text contains explicit profanity or hostile curses."""
    if not text:
        return False
    return bool(PROFANITY_PATTERN.search(text))


TECHNICAL_ACRONYMS = {
    "API", "REST", "JSON", "HTTP", "HTTPS", "HTML", "CSS", "SQL", "URL",
    "SDK", "GPT", "CPU", "GPU", "RAM", "AWS", "XML", "SSH", "TCP", "UDP",
    "IP", "DNS", "LLM", "CSV", "PDF", "USA", "UK", "ID", "AI", "ML", "OS",
}


def detect_aggressive_caps(text: str, min_caps_words: int = 2, min_length: int = 8) -> bool:
    """Detect aggressive ALL CAPS usage.

    Requires at least min_caps_words non-acronym words in ALL CAPS (length >= 3)
    and an uppercase ratio > 0.45 among alphabetic characters.
    """
    if not text or len(text.strip()) < min_length:
        return False

    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 6:
        return False

    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    words = re.findall(r"\b[A-Z]{3,}\b", text)

    # Exclude technical and standard acronyms
    shouting_words = [w for w in words if w not in TECHNICAL_ACRONYMS]

    return len(shouting_words) >= min_caps_words and upper_ratio >= 0.45


def detect_negative_cues(text: str) -> bool:
    """Check if text contains explicit negative feedback, correction cues, or aggressive punctuation."""
    if not text:
        return False
    return bool(NEGATIVE_CUES_PATTERN.search(text))


def extract_rule_signals(u_t: str, u_next: str) -> Dict[str, Any]:
    """Extract all deterministic rule-based signals from user reaction U_{t+1} and prompt U_t.

    Returns dict containing:
      - is_repeat_prompt: bool
      - is_batch_task: bool
      - has_profanity: bool
      - has_aggressive_caps: bool
      - has_negative_cues: bool
      - rule_dissatisfaction: bool
    """
    is_batch = detect_batch_task(u_t, u_next)
    is_repeat = detect_repeat_prompt(u_t, u_next)

    # If it is a batch task, it is NOT an irritated repetition
    if is_batch:
        is_repeat = False

    profanity = detect_profanity(u_next)
    caps = detect_aggressive_caps(u_next)
    negative_cues = detect_negative_cues(u_next)

    rule_dissat = (profanity or caps or negative_cues or is_repeat) and not is_batch

    return {
        "is_repeat_prompt": bool(is_repeat),
        "is_batch_task": bool(is_batch),
        "has_profanity": bool(profanity),
        "has_aggressive_caps": bool(caps),
        "has_negative_cues": bool(negative_cues),
        "rule_dissatisfaction": bool(rule_dissat),
    }
