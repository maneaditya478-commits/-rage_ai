"""Pretrained emotion inference using DistilRoBERTa on U_{t+1}."""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure offline loading from local Hugging Face cache
os.environ["HF_HUB_OFFLINE"] = "1"


class EmotionClassifier:
    """Inference wrapper for j-hartmann/emotion-english-distilroberta-base."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        self.model.eval()

        # Labels: {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}
        self.id2label = self.model.config.id2label
        self.anger_idx = 0
        for idx, lbl in self.id2label.items():
            if lbl.lower() == "anger":
                self.anger_idx = int(idx)
                break

    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Run batched inference on a list of texts with length bucketing for speed.

        Truncates each text to max_length (512 tokens).

        Returns:
            Tuple of:
              - anger_probs: float array of anger probabilities [0.0, 1.0]
              - dominant_emotions: list of string emotion names
        """
        n_samples = len(texts)
        if n_samples == 0:
            return np.array([], dtype=np.float32), []

        # Sort indices by length so batches have uniform lengths, minimizing padding overhead
        sorted_indices = sorted(range(n_samples), key=lambda idx: len(texts[idx] or ""))
        results_anger: List[float] = [0.0] * n_samples
        results_dominant: List[str] = [""] * n_samples

        total_batches = (n_samples + self.batch_size - 1) // self.batch_size
        pbar = tqdm(
            range(0, n_samples, self.batch_size),
            total=total_batches,
            desc="Emotion RoBERTa",
            file=sys.stdout,
        )

        for b_start in pbar:
            batch_idxs = sorted_indices[b_start : b_start + self.batch_size]
            batch_texts = [texts[idx] for idx in batch_idxs]
            safe_texts = [t if t and t.strip() else "..." for t in batch_texts]

            inputs = self.tokenizer(
                safe_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.inference_mode():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            anger_p = probs[:, self.anger_idx]
            top_indices = np.argmax(probs, axis=-1)

            for orig_idx, ap, ti in zip(batch_idxs, anger_p, top_indices):
                results_anger[orig_idx] = float(ap)
                results_dominant[orig_idx] = self.id2label[ti]

        return np.array(results_anger, dtype=np.float32), results_dominant
