"""Label generation and aggregation for TokenRage.

Builds two primary ground-truth labels:
1. strict_anger: High RoBERTa anger score (>= 0.40 or dominant emotion == 'anger')
   OR explicit profanity / aggressive ALL CAPS.
2. dissatisfaction: Broader frustration metric including strict_anger, repeated questions,
   and negative feedback cues ("wrong", "again", "stop", "I told you", "?!").

Crucially excludes batch tasks (shared templates with different payloads) from frustration.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def create_ground_truth_labels(
    df: pd.DataFrame,
    anger_threshold: float = 0.40,
) -> pd.DataFrame:
    """Combine emotion predictions and rule signals into strict_anger and dissatisfaction labels.

    Args:
        df: DataFrame containing emotion and rule columns:
            - anger_prob
            - dominant_emotion
            - has_profanity
            - has_aggressive_caps
            - has_negative_cues
            - is_repeat_prompt
            - is_batch_task
            - is_long_pasted_content
        anger_threshold: Probability threshold for emotion anger. Default: 0.40.

    Returns:
        DataFrame with added target labels: 'strict_anger' and 'dissatisfaction'.
    """
    df = df.copy()

    # 1. Base emotion trigger
    model_anger = (df["anger_prob"] >= anger_threshold) | (df["dominant_emotion"] == "anger")

    # 2. Strict anger: Emotion anger OR profanity OR aggressive CAPS
    # Exclude batch tasks
    strict_anger_cond = (
        (model_anger | df["has_profanity"].astype(bool) | df["has_aggressive_caps"].astype(bool))
        & (~df["is_batch_task"].astype(bool))
    )
    df["strict_anger"] = strict_anger_cond.astype(int)

    # 3. Dissatisfaction: strict_anger OR repeat prompt OR negative cues ("wrong", "again", "stop", "?!")
    # Exclude batch tasks
    dissatisfaction_cond = (
        (
            strict_anger_cond
            | df["is_repeat_prompt"].astype(bool)
            | df["has_negative_cues"].astype(bool)
        )
        & (~df["is_batch_task"].astype(bool))
    )
    df["dissatisfaction"] = dissatisfaction_cond.astype(int)

    return df


def compute_label_distribution_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute detailed label positive rates and covariate breakdowns."""
    n_total = len(df)
    strict_pos = df["strict_anger"].sum()
    dissat_pos = df["dissatisfaction"].sum()
    batch_cnt = df["is_batch_task"].sum()
    repeat_cnt = df["is_repeat_prompt"].sum()

    report: Dict[str, Any] = {
        "total_samples": n_total,
        "strict_anger_count": int(strict_pos),
        "strict_anger_rate": float(strict_pos / n_total),
        "dissatisfaction_count": int(dissat_pos),
        "dissatisfaction_rate": float(dissat_pos / n_total),
        "batch_task_count": int(batch_cnt),
        "batch_task_rate": float(batch_cnt / n_total),
        "repeat_prompt_count": int(repeat_cnt),
        "repeat_prompt_rate": float(repeat_cnt / n_total),
    }

    # Model Breakdown
    if "is_gpt4" in df.columns:
        df_gpt4 = df[df["is_gpt4"] == 1]
        df_gpt35 = df[df["is_gpt4"] == 0]
        report["by_model"] = {
            "gpt_3.5_turbo": {
                "n": len(df_gpt35),
                "strict_anger_rate": float(df_gpt35["strict_anger"].mean()) if len(df_gpt35) > 0 else 0.0,
                "dissatisfaction_rate": float(df_gpt35["dissatisfaction"].mean()) if len(df_gpt35) > 0 else 0.0,
            },
            "gpt_4": {
                "n": len(df_gpt4),
                "strict_anger_rate": float(df_gpt4["strict_anger"].mean()) if len(df_gpt4) > 0 else 0.0,
                "dissatisfaction_rate": float(df_gpt4["dissatisfaction"].mean()) if len(df_gpt4) > 0 else 0.0,
            },
        }

    # Topic Shift Breakdown
    if "topic_shift" in df.columns:
        df_shift = df[df["topic_shift"] == True]
        df_noshift = df[df["topic_shift"] == False]
        report["by_topic_shift"] = {
            "with_topic_shift": {
                "n": len(df_shift),
                "strict_anger_rate": float(df_shift["strict_anger"].mean()) if len(df_shift) > 0 else 0.0,
                "dissatisfaction_rate": float(df_shift["dissatisfaction"].mean()) if len(df_shift) > 0 else 0.0,
            },
            "without_topic_shift": {
                "n": len(df_noshift),
                "strict_anger_rate": float(df_noshift["strict_anger"].mean()) if len(df_noshift) > 0 else 0.0,
                "dissatisfaction_rate": float(df_noshift["dissatisfaction"].mean()) if len(df_noshift) > 0 else 0.0,
            },
        }

    return report
