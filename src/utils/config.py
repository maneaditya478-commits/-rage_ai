"""Configuration settings and constants for TokenRage."""

from dataclasses import dataclass
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Project-wide hyperparameters and configuration settings."""
    # Reproducibility
    random_seed: int = 42

    # Dataset settings
    dataset_name: str = "allenai/WildChat"
    dataset_shard_file: str = "data/train-00000-of-00006.parquet"
    target_sample_size: int = 50_000
    language: str = "English"
    min_turns: int = 2  # At least 2 turns (User -> Assistant -> User -> ...)
    
    # Tiktoken tokenizer
    tiktoken_encoding: str = "cl100k_base"

    # Emotion model
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    batch_size_inference: int = 64

    # Train/Validation/Test split ratios (Conversation-level)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


CONFIG = PipelineConfig()
