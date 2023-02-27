#!/usr/bin/env python
import os

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support as score

from util import get_codename, parse_args, sample, ENCODE_PHENOTYPE


def main() -> None:
    """Runs Pre-trained Siamese Neural Network (PT-SNN) experiments and logs the results."""

    args = parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codename = get_codename(args.model)
    model = SentenceTransformer(model_name_or_path=codename, device=device)

    # Stores the generated output/prediction tensors for analyses
    to_save = []

    # We encode as [precision, recall, fscore] hence the second dimension is 3
    metrics = torch.zeros(args.eval_iter, 3)
    for idx in range(args.eval_iter):
        # Uses `idx` as the random seed: builds pseudo-random datasets in a reproducible manner
        df_train, df_test = sample(args.path, args.shots, random_state=idx)

        # Generates embeddings using transformer models and convert to PyTorch tensors
        emb_train = torch.tensor(df_train["text"].apply(lambda s: model.encode(s)).array).to(device)
        emb_test = torch.tensor(df_test["text"].apply(lambda s: model.encode(s)).array).to(device)

        # Encodes the labels and convert to PyTorch tensors
        lbl_train = torch.tensor(df_train["label"].apply(lambda l: ENCODE_PHENOTYPE[l]).array)
        lbl_test = torch.tensor(df_test["label"].apply(lambda l: ENCODE_PHENOTYPE[l]).array)

        # L2-normalizes the embeddings and computes the cosine similarity table
        similarity_table = F.normalize(emb_test) @ F.normalize(emb_train).T

        # Groups by label and aggregates by label and generates predictions
        label_table = torch.zeros(lbl_train.max() + 1, lbl_train.numel()).to(device)
        label_table[lbl_train, torch.arange(lbl_train.numel())] = 1
        label_table = F.normalize(label_table, p=1, dim=1)
        out = (similarity_table @ label_table.T).argmax(dim=1).cpu()

        precision, recall, fscore, _ = score(lbl_test, out, average="weighted")
        metrics[idx] = torch.tensor([precision, recall, fscore])

        to_save.append(out.tolist())

    precision, recall, fscore = metrics.mean(dim=0).tolist()
    print(f"Precision: {precision:.2}")
    print(f"Recall:    {recall:.2}")
    print(f"F-score:   {fscore:.2}")

    filename = f"{args.log_dir}_{args.model}_{args.shots}.pt"
    torch.save(torch.tensor(list(zip(*to_save))), os.path.join(args.log_dir, filename))


if __name__ == "__main__":
    main()
