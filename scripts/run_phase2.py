"""Phase 2 Pipeline: Feature Extraction, Topic Shift & Batch Task Detection,
Emotion Scoring, and Combined Frustration Labeling.

Executes:
1. Load processed conversational pairs
2. Extract linguistic and structural features (tiktoken tokens, ratios, repetition, apologies)
3. Detect repeated prompts, batch tasks, and topic shifts (all-MiniLM-L6-v2)
4. Run pretrained emotion classification (j-hartmann/emotion-english-distilroberta-base)
5. Generate 'strict_anger' and 'dissatisfaction' labels (with batch tasks excluded)
6. Compute detailed distribution metrics (by model, by topic shift)
7. Save full labeled dataset and export 30 random examples for manual review
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Offline HF hub usage
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extraction import extract_features_for_dataframe
from src.features.repetition_and_similarity import TopicShiftDetector
from src.labeling.combiner import compute_label_distribution_report, create_ground_truth_labels
from src.labeling.emotion_model import EmotionClassifier
from src.labeling.rules import extract_rule_signals
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, REPORTS_DIR


def run_phase2(sample_size: int = 10_000):
    print("=" * 80)
    print("TOKENRAGE: PHASE 2 - FEATURE EXTRACTION & FRUSTRATION LABELING PIPELINE")
    print("=" * 80)

    # 1. Load paired transitions
    input_parquet = PROCESSED_DATA_DIR / "conversation_pairs.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing {input_parquet}. Run scripts/run_phase1.py first.")

    print(f"Loading paired transitions from {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    print(f"Total available pairs: {len(df):,}")

    # Subsample conversations for Phase 2 if requested
    if sample_size and len(df) > sample_size:
        # Sample by unique conversations to preserve conversational continuity
        unique_convs = df["conversation_id"].unique()
        np.random.seed(CONFIG.random_seed)
        # Calculate approx convs needed
        avg_turns = len(df) / len(unique_convs)
        n_convs_needed = int(sample_size / avg_turns)
        sampled_convs = np.random.choice(unique_convs, size=min(n_convs_needed, len(unique_convs)), replace=False)
        df = df[df["conversation_id"].isin(sampled_convs)].copy().reset_index(drop=True)
        if len(df) > sample_size:
            df = df.iloc[:sample_size].copy().reset_index(drop=True)
        print(f"Sampled {len(df):,} pairs across {df['conversation_id'].nunique():,} unique conversations.")

    # 2. Extract linguistic and structural features
    print("\n[Step 1/5] Extracting linguistic, structural, and repetition features...")
    df = extract_features_for_dataframe(df)

    # 3. Rule signals & Repeat / Batch task detection
    print("\n[Step 2/5] Extracting deterministic rule signals (repeat prompt, batch task, profanity, CAPS, cues)...")
    rule_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rules"):
        u_t = row["user_prompt"]
        u_next = row["next_user_reaction"]
        signals = extract_rule_signals(u_t, u_next)
        rule_records.append(signals)

    rule_df = pd.DataFrame(rule_records)
    df = pd.concat([df, rule_df], axis=1)

    # 4. Sentence Transformer Topic Shift Detection (all-MiniLM-L6-v2)
    print("\n[Step 3/5] Computing sentence embedding topic shifts (all-MiniLM-L6-v2)...")
    topic_detector = TopicShiftDetector(threshold=0.35)
    similarities, topic_shifts = topic_detector.compute_similarity_batch(
        prompts_t=df["user_prompt"].tolist(),
        prompts_next=df["next_user_reaction"].tolist(),
        batch_size=128,
    )
    df["semantic_similarity_prompts"] = similarities
    df["topic_shift"] = topic_shifts.astype(int)

    # 5. Pretrained Emotion Model Inference (DistilRoBERTa on U_{t+1})
    print("\n[Step 4/5] Running RoBERTa emotion classification on U_{t+1} (512 token truncation)...")
    emotion_clf = EmotionClassifier(batch_size=CONFIG.batch_size_inference, max_length=512)
    anger_probs, dominant_emotions = emotion_clf.predict_batch(df["next_user_reaction"].tolist())
    df["anger_prob"] = anger_probs
    df["dominant_emotion"] = dominant_emotions

    # 6. Combined Ground-Truth Labeling
    print("\n[Step 5/5] Generating ground-truth labels (strict_anger & dissatisfaction)...")
    df = create_ground_truth_labels(df, anger_threshold=0.40)

    # 7. Distribution Metrics & Reporting
    report = compute_label_distribution_report(df)

    print("\n" + "=" * 50)
    print("PHASE 2 LABEL DISTRIBUTION REPORT")
    print("=" * 50)
    print(f"Total Transition Pairs Evaluated : {report['total_samples']:,}")
    print(f"Batch Tasks Detected (Excluded)  : {report['batch_task_count']:,} ({report['batch_task_rate']*100:.2f}%)")
    print(f"Repeated Prompts Detected        : {report['repeat_prompt_count']:,} ({report['repeat_prompt_rate']*100:.2f}%)")
    print("-" * 50)
    print(f"Strict Anger Positive Count      : {report['strict_anger_count']:,} ({report['strict_anger_rate']*100:.2f}%)")
    print(f"Dissatisfaction Positive Count   : {report['dissatisfaction_count']:,} ({report['dissatisfaction_rate']*100:.2f}%)")
    print("-" * 50)

    print("\n--- Breakdown By Model ---")
    for model_name, stats in report.get("by_model", {}).items():
        print(f"  [{model_name.upper()}] (n={stats['n']:,}):")
        print(f"    Strict Anger Rate    : {stats['strict_anger_rate']*100:.2f}%")
        print(f"    Dissatisfaction Rate : {stats['dissatisfaction_rate']*100:.2f}%")

    print("\n--- Breakdown By Topic Shift ---")
    for shift_name, stats in report.get("by_topic_shift", {}).items():
        print(f"  [{shift_name.upper()}] (n={stats['n']:,}):")
        print(f"    Strict Anger Rate    : {stats['strict_anger_rate']*100:.2f}%")
        print(f"    Dissatisfaction Rate : {stats['dissatisfaction_rate']*100:.2f}%")

    # Save labeled dataset
    output_labeled_path = PROCESSED_DATA_DIR / "labeled_dataset.parquet"
    df.to_parquet(output_labeled_path, index=False)
    print(f"\nSaved full labeled dataset ({len(df):,} rows) to: {output_labeled_path}")

    # Save distribution report to JSON
    report_json_path = REPORTS_DIR / "phase2_label_distribution.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved distribution report to: {report_json_path}")

    # 8. Export 30 Random Examples for Manual Review
    print("\n" + "=" * 50)
    print("30 RANDOM LABELED EXAMPLES FOR MANUAL REVIEW")
    print("=" * 50)
    samples_30 = df.sample(n=30, random_state=CONFIG.random_seed).reset_index(drop=True)

    export_cols = [
        "conversation_id",
        "turn_idx",
        "model",
        "tokens_assistant",
        "token_ratio",
        "is_batch_task",
        "is_repeat_prompt",
        "topic_shift",
        "anger_prob",
        "dominant_emotion",
        "has_negative_cues",
        "has_profanity",
        "has_aggressive_caps",
        "strict_anger",
        "dissatisfaction",
        "user_prompt",
        "assistant_reply",
        "next_user_reaction",
    ]
    samples_export = samples_30[export_cols].copy()
    samples_csv_path = PROCESSED_DATA_DIR / "sample_30_labeled_examples.csv"
    samples_export.to_csv(samples_csv_path, index=False, encoding="utf-8")
    print(f"Saved 30 manual review samples to: {samples_csv_path}\n")

    for idx, row in samples_30.iterrows():
        print(f"[{idx+1:02d}] Conv: {row['conversation_id'][:8]} | Turn: {row['turn_idx']} | Model: {row['model']}")
        print(f"     Tokens AI: {row['tokens_assistant']} | Token Ratio: {row['token_ratio']:.1f}x | BatchTask: {row['is_batch_task']} | TopicShift: {bool(row['topic_shift'])}")
        print(f"     RoBERTa Anger: {row['anger_prob']:.3f} ({row['dominant_emotion']}) | NegCues: {row['has_negative_cues']} | Profanity: {row['has_profanity']} | CAPS: {row['has_aggressive_caps']}")
        print(f"     >>> LABELS -> Strict Anger: {row['strict_anger']} | Dissatisfaction: {row['dissatisfaction']}")
        print(f"     U_t   : {row['user_prompt'][:100]}..." if len(row['user_prompt']) > 100 else f"     U_t   : {row['user_prompt']}")
        print(f"     A_t   : {row['assistant_reply'][:120]}..." if len(row['assistant_reply']) > 120 else f"     A_t   : {row['assistant_reply']}")
        print(f"     U_t+1 : {row['next_user_reaction'][:120]}..." if len(row['next_user_reaction']) > 120 else f"     U_t+1 : {row['next_user_reaction']}")
        print("-" * 70)

    print("\nPhase 2 Pipeline Successfully Completed!")
    return df, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=10_000, help="Number of pairs to process (default: 10000)")
    args = parser.parse_args()
    run_phase2(sample_size=args.sample_size)
