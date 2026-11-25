"""Prompt similarity, repeat prompt detection, batch task detection, and topic shift."""

import os
import re
from typing import List, Tuple

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# Ensure Hugging Face cache is used offline in sandboxed/production runs
os.environ["HF_HUB_OFFLINE"] = "1"

# Shared prefix minimum characters to suspect template/batch behavior
MIN_TEMPLATE_PREFIX_LEN = 25


def detect_repeat_prompt(u_t: str, u_next: str, ratio_threshold: float = 90.0) -> bool:
    """Detect whether U_{t+1} is essentially a repetition of U_t using full-text similarity.

    Args:
        u_t: User prompt at turn t.
        u_next: User prompt at turn t+1.
        ratio_threshold: Rapidfuzz full-string ratio threshold (0-100). Default: 90.0.

    Returns:
        True if the prompts are essentially identical repetitions.
    """
    if not u_t or not u_next:
        return False
    u_t_clean = u_t.strip().lower()
    u_next_clean = u_next.strip().lower()

    # Exact match or full-text Levenshtein ratio >= threshold
    if u_t_clean == u_next_clean:
        return True

    ratio = fuzz.ratio(u_t_clean, u_next_clean)
    return ratio >= ratio_threshold


def detect_batch_task(u_t: str, u_next: str) -> bool:
    """Detect if U_t and U_{t+1} are part of a batch task (shared template, different content).

    Examples of batch tasks:
      U_t:    "Translate the following sentence to French: The book is on the table."
      U_{t+1}: "Translate the following sentence to French: She walked to the market."
      U_t:    "Fix any punctuation errors below: [Miguel Rivera loves Mickey Mouse...]"
      U_{t+1}: "Fix any punctuation errors below: [Tony Stark's parrot can...]"

    In these cases, high lexical or partial overlap reflects a batch workflow, NOT user anger.
    """
    if not u_t or not u_next:
        return False

    u_t_clean = u_t.strip().lower()
    u_next_clean = u_next.strip().lower()

    # If they are virtually identical (>95%), that's a true repeat, not a batch task
    full_ratio = fuzz.ratio(u_t_clean, u_next_clean)
    if full_ratio >= 95.0:
        return False

    # Find common prefix length
    min_len = min(len(u_t_clean), len(u_next_clean))
    prefix_len = 0
    for i in range(min_len):
        if u_t_clean[i] == u_next_clean[i]:
            prefix_len += 1
        else:
            break

    # If they share a substantial template prefix (>= 25 chars or >= 35% of length)
    # but the full similarity is between 35% and 94% (differing payloads)
    if prefix_len >= MIN_TEMPLATE_PREFIX_LEN and 35.0 <= full_ratio < 95.0:
        return True

    # Check common template suffix (e.g., "...output in json format", "...explain your reasoning")
    suffix_len = 0
    for i in range(1, min_len + 1):
        if u_t_clean[-i] == u_next_clean[-i]:
            suffix_len += 1
        else:
            break
    if suffix_len >= MIN_TEMPLATE_PREFIX_LEN and 35.0 <= full_ratio < 95.0:
        return True

    # Token set ratio high (template overlap) while token sort / full ratio is distinct
    token_set = fuzz.token_set_ratio(u_t_clean, u_next_clean)
    if token_set >= 85.0 and full_ratio < 80.0 and len(u_t_clean) >= 30 and len(u_next_clean) >= 30:
        return True

    return False


class TopicShiftDetector:
    """Computes sentence-embedding cosine similarities using all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.35,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def compute_similarity_batch(
        self,
        prompts_t: List[str],
        prompts_next: List[str],
        batch_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarity between pairs of prompts.

        Returns:
            Tuple of:
              - similarities: float array of cosine similarities [-1.0, 1.0]
              - topic_shifts: boolean array indicating whether similarity < threshold
        """
        # Encode both sets
        emb_t = self.model.encode(
            prompts_t,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb_next = self.model.encode(
            prompts_next,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Dot product of normalized vectors = cosine similarity
        similarities = np.sum(emb_t * emb_next, axis=1)
        topic_shifts = similarities < self.threshold
        return similarities, topic_shifts
