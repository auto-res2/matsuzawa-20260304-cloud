"""
Data preprocessing for GSM8K dataset.
"""

import os
from datasets import load_dataset
from typing import List, Dict


def load_gsm8k(
    split: str = "test", num_samples: int = None, cache_dir: str = ".cache"
) -> List[Dict]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train/test)
        num_samples: Number of samples to load (None for all)
        cache_dir: Cache directory for datasets

    Returns:
        List of examples with 'question' and 'answer' fields
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list
    examples = []
    for i, example in enumerate(dataset):
        if num_samples is not None and i >= num_samples:
            break

        # Extract numeric answer from the "#### X" format
        answer_text = example["answer"]
        numeric_answer = extract_numeric_answer(answer_text)

        examples.append(
            {
                "question": example["question"],
                "answer": numeric_answer,
                "full_answer": answer_text,
            }
        )

    return examples


def extract_numeric_answer(answer_text: str) -> float:
    """
    Extract numeric answer from GSM8K answer format.
    GSM8K answers end with "#### X" where X is the numeric answer.

    Args:
        answer_text: Full answer text

    Returns:
        Numeric answer as float
    """
    if "####" in answer_text:
        answer_str = answer_text.split("####")[-1].strip()
        # Remove commas and parse
        answer_str = answer_str.replace(",", "")
        try:
            return float(answer_str)
        except ValueError:
            # If parsing fails, return NaN
            return float("nan")
    return float("nan")
