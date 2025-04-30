import torch
import transformers
from datasets import load_dataset

def show_gsm8k_samples():
    """
    Display first 5 samples from the GSM8K dataset.
    """
    dataset = load_dataset("gsm8k", "main", split="train")
    samples = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    for i in range(5):
        print(f"Question: {samples[i]['question']}")
        print(f"Answer: {samples[i]['answer']}")
        print()