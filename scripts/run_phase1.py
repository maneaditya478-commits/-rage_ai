"""Phase 1 Script: Repository setup, dataset download/loading, sanitization, and basic statistics.

Executes:
1. Verification of environment and paths
2. Dataset loading from Hugging Face WildChat
3. Filtering for English multi-turn conversations
4. PII scrubbing & extraction of (User_t, Assistant_t, User_{t+1}) pairs
5. Computation of summary statistics
6. Display of 5 representative sample rows
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import (
    download_or_locate_dataset,
    extract_turn_pairs,
    load_raw_dataset,
    save_processed_pairs,
)
from src.utils.config import CONFIG, PROCESSED_DATA_DIR


def run_phase1():
    print("=" * 80)
    print("TOKENRAGE: PHASE 1 - REPO SETUP, DATASET INGESTION & BASIC STATS")
    print("=" * 80)

    # 1. Download or locate dataset shard
    parquet_path = download_or_locate_dataset()

    # 2. Filter raw dataset
    conv_df = load_raw_dataset(
        parquet_path=parquet_path,
        target_sample_size=CONFIG.target_sample_size,
        language=CONFIG.language,
        min_turns=CONFIG.min_turns,
    )

    # 3. Extract transition pairs (Assistant reply -> Next user reaction)
    pairs_df = extract_turn_pairs(conv_df)

    # 4. Save raw pairs to processed data folder
    output_path = save_processed_pairs(pairs_df)

    # 5. Basic Statistics
    print("\n" + "=" * 50)
    print("BASIC STATISTICAL SUMMARY")
    print("=" * 50)
    print(f"Total Conversations Ingested: {len(conv_df):,}")
    print(f"Total (Assistant reply -> Next user reaction) Pairs: {len(pairs_df):,}")
    print(f"Average transitions per conversation: {len(pairs_df) / len(conv_df):.2f}")

    print("\n--- Model Distribution across pairs ---")
    model_counts = pairs_df["model"].value_counts()
    for model, count in model_counts.items():
        print(f"  {model:20s}: {count:,} ({count / len(pairs_df) * 100:.1f}%)")

    print("\n--- Turn Index Distribution ---")
    print(pairs_df["turn_idx"].describe())

    # Text length summaries (character and approximate word counts)
    pairs_df["user_prompt_chars"] = pairs_df["user_prompt"].str.len()
    pairs_df["assistant_reply_chars"] = pairs_df["assistant_reply"].str.len()
    pairs_df["next_reaction_chars"] = pairs_df["next_user_reaction"].str.len()

    print("\n--- Assistant Reply Character Length Distribution ---")
    print(pairs_df["assistant_reply_chars"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99]))

    print("\n--- Next User Reaction Character Length Distribution ---")
    print(pairs_df["next_reaction_chars"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99]))

    # 6. Show 5 Sample Rows
    print("\n" + "=" * 50)
    print("5 REPRESENTATIVE SAMPLE ROWS")
    print("=" * 50)
    samples = pairs_df.sample(n=5, random_state=CONFIG.random_seed).reset_index(drop=True)
    for idx, row in samples.iterrows():
        print(f"\n[Sample #{idx + 1}]")
        print(f"Conversation ID : {row['conversation_id']}")
        print(f"Model           : {row['model']}")
        print(f"Turn Index      : {row['turn_idx']} (of {row['total_turns']} total turns)")
        print(f"User Prompt     : {row['user_prompt'][:120]}..." if len(row['user_prompt']) > 120 else f"User Prompt     : {row['user_prompt']}")
        print(f"Assistant Reply : {row['assistant_reply'][:160]}..." if len(row['assistant_reply']) > 160 else f"Assistant Reply : {row['assistant_reply']}")
        print(f"Next Reaction   : {row['next_user_reaction'][:120]}..." if len(row['next_user_reaction']) > 120 else f"Next Reaction   : {row['next_user_reaction']}")
        print("-" * 50)

    # Save a small sample CSV for quick viewing and manual inspection
    sample_csv_path = PROCESSED_DATA_DIR / "sample_5_rows.csv"
    samples.to_csv(sample_csv_path, index=False)
    print(f"\nSaved 5 sample rows to {sample_csv_path}")

    print("\nPhase 1 Completed Successfully.")


if __name__ == "__main__":
    run_phase1()
