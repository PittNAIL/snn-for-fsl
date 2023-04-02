#!/usr/bin/env python
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from sklearn.metrics import precision_recall_fscore_support as score

from util import gpt2_annotations


def main():
    """Evaluates Generative Pre-trained Transformer 2 (GPT-2)."""

    print("Benchmarking Pre-trained Transformer 2 (GPT-2)...\n")

    print("GPT-2 (4 Shots)\n================================")
    df = gpt2_annotations(shots=4)
    precision, recall, fscore, _ = score(df["correct"], df["most likely class"], average="weighted")
    print(f"Precision: {precision:.2}")
    print(f"Recall:    {recall:.2}")
    print(f"F-score:   {fscore:.2}")

    print("\nGPT-2 (8 Shots)\n================================")
    df = gpt2_annotations(shots=8)
    precision, recall, fscore, _ = score(df["correct"], df["most likely class"], average="weighted")
    print(f"Precision: {precision:.2}")
    print(f"Recall:    {recall:.2}")
    print(f"F-score:   {fscore:.2}")

    print("\nGPT-2 (16 Shots)\n================================")
    df = gpt2_annotations(shots=16)
    precision, recall, fscore, _ = score(df["correct"], df["most likely class"], average="weighted")
    print(f"Precision: {precision:.2}")
    print(f"Recall:    {recall:.2}")
    print(f"F-score:   {fscore:.2}")

    print("\nDone!")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
