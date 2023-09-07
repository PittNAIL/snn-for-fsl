import argparse

import pandas as pd


ENCODE_PHENOTYPE = {
    "ADVANCED.CANCER": 0,
    "ADVANCED.HEART.DISEASE": 1,
    "ADVANCED.LUNG.DISEASE": 2,
    "CHRONIC.PAIN.FIBROMYALGIA": 3,
}


def generate_training_triplets(df_train: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Generates triplets for training SNNs."""

    triplets = []
    seen = set()
    for _, (text1, label1, emb1, lbl1) in df_train.iterrows():
        for _, (text2, label2, emb2, lbl2) in df_train.iterrows():
            if text1 == text2 and lbl1 == lbl2:
                continue

            if (text1, text2) in seen or (text2, text1) in seen:
                continue

            seen.add((text1, text2))
            triplets.append((emb1, emb2, int(label1 == label2)))

    return triplets


def get_codename(model: str) -> str:
    """Gets the codename by the model name."""

    if model == "bert":
        return "bert-base-cased"

    if model == "biobert":
        return "dmis-lab/biobert-base-cased-v1.2"

    if model == "bioclinicalbert":
        return "emilyalsentzer/Bio_ClinicalBERT"

    if model == "gpt2":
        return "gpt2"

    raise ValueError(f"Unknown model: {model}")


def gpt2_annotations(shots: int) -> pd.DataFrame:
    """Obtains GPT-2 annotations for the given number of shots."""

    options = ", ".join(" ".join(val.lower().split(".")) for val in ENCODE_PHENOTYPE.keys())
    prompt = f"options are {options}. type of disease"

    _, df_test = sample("phenotype.csv", shots, random_state=0)
    df_test.reset_index(inplace=True)

    prompt_fn = lambda s: f"{s[:1_024 - len(prompt) - 2]}. {prompt}"
    df_test["question"] = df_test["text"].map(prompt_fn)

    annotations = pd.read_csv("gpt2-medium-annotations.csv")
    annotations.reset_index(inplace=True)

    df = annotations.loc[annotations["question"].isin(df_test["question"])]

    return df


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Pretrained SNN Benchmarking")
    parser.add_argument("--model", type=str, help="model type", required=True)
    parser.add_argument("--shots", type=int, help="number of shots", required=True)
    parser.add_argument("--path", type=str, help="path to the dataset", required=True)
    parser.add_argument("--eval_iter", type=int, help="iterations for averaging", required=True)
    parser.add_argument("--log_dir", type=str, help="logging directory", required=True)

    return parser.parse_args()


def sample(path: str, num_samples: int, random_state: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Samples the dataset equally per class."""

    # Read and shuffle
    df = pd.read_csv(path).sample(frac=1, random_state=random_state)

    # Train/test split where training set is compirised of n samples per classes
    train = df.groupby("label", group_keys=False).apply(
        lambda label: label.sample(num_samples, random_state=random_state)
    )
    test = df.drop(train.index)

    return train, test
