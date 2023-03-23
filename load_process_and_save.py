#!/usr/bin/env python
import pandas as pd


def load_process_and_save(path: str, save_path: str) -> None:
    """Loads, processes, and saves the data frame."""

    question, response, correct = [], [], []

    df = pd.read_csv(path)
    for _, row in df.iterrows():
        q, r, c = row
        question.append(q)
        response.append(r.replace(q, ""))
        correct.append(c)

    new_df = pd.DataFrame({"question": question, "response": response, "correct": correct})
    new_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    load_process_and_save("gpt2-medium/gpt2_gpt2_4_0.csv", "gpt2-medium-to-annotate.csv")
