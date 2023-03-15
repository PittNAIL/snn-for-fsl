#!/usr/bin/env python
import os

import torch

from transformers import pipeline, set_seed

from util import parse_args, sample, ENCODE_PHENOTYPE


# Sets the random seed for reproducible experiments
set_seed(1337)


def main():
    """Runs Generative Pre-trained Transformer 2 (GPT-2) and reports the statistics."""

    args = parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    gpt2 = pipeline("text-generation", model="gpt2-medium", device=device)

    options = ", ".join(" ".join(val.lower().split(".")) for val in ENCODE_PHENOTYPE.keys())
    prompt = f"options are {options}. type of disease"
    for idx in range(args.eval_iter):
        # Uses `idx` as the random seed: builds pseudo-random datasets in a reproducible manner
        _, df_test = sample(args.path, args.shots, random_state=idx)

        # Generates GPT-2 responses
        prompt_fn = lambda s: f"{s[:1_024 - len(prompt) - 2]}. {prompt}"
        df_test["question"] = df_test["text"].map(prompt_fn)
        df_test["response"] = df_test["question"].map(lambda q: gpt2(q, max_length=1_024)[0]["generated_text"])
        df_test["correct"] = df_test["label"].map(lambda l: " ".join(l.lower().split(".")))

        # Saves the results for human evaluation
        filename = f"{args.log_dir}_{args.model}_{args.shots}_{idx}.csv"
        df_test[["question", "response", "correct"]].to_csv(
            os.path.join(args.log_dir, filename), sep=",", index=False
        )


if __name__ == "__main__":
    main()
