#!/usr/bin/env python
import os
import sys

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support as score

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from util import sample, ENCODE_PHENOTYPE


def k_shot_eval_annotations(shots: int) -> None:
    """Performs K-shot evaluation."""

    options = ", ".join(" ".join(val.lower().split(".")) for val in ENCODE_PHENOTYPE.keys())
    prompt = f"options are {options}. type of disease"

    _, df_test = sample("phenotype.csv", shots, random_state=0)
    df_test.reset_index(inplace=True)

    prompt_fn = lambda s: f"{s[:1_024 - len(prompt) - 2]}. {prompt}"
    df_test["question"] = df_test["text"].map(prompt_fn)

    annotations = pd.read_csv("gpt2-medium-annotations.csv")
    annotations.reset_index(inplace=True)

    df = annotations.loc[annotations["question"].isin(df_test["question"])]

    precision, recall, fscore, _ = score(df["correct"], df["most likely class"], average="weighted")
    print(f"Precision: {precision:.2}")
    print(f"Recall:    {recall:.2}")
    print(f"F-score:   {fscore:.2}")


def main():
    """Evaluates Generative Pre-trained Transformer 2 (GPT-2)."""

    print("Benchmarking Pre-trained Transformer 2 (GPT-2)...\n")

    print("GPT-2 (4 Shots)\n================================")
    k_shot_eval_annotations(shots=4)

    print("\nGPT-2 (8 Shots)\n================================")
    k_shot_eval_annotations(shots=8)

    print("\nGPT-2 (16 Shots)\n================================")
    k_shot_eval_annotations(shots=16)

    print("\nDone!")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
