#!/usr/bin/env python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support as score
from transformers import set_seed

from util import generate_training_triplets, get_codename, parse_args, sample, ENCODE_PHENOTYPE


# Sets the random seed for reproducible experiments
set_seed(1337)


class SNNTrainDataset(torch.utils.data.Dataset):
    """Dataset for training SNNs."""

    def __init__(self, df_train) -> None:
        self.data = generate_training_triplets(df_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class SiameseNeuralNetwork(nn.Module):
    """Siamese Neural Network (SNN) model."""

    def __init__(self, dataloader) -> None:
        """Initializes the model."""

        super().__init__()

        self.dataloader = dataloader

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.22,
            bidirectional=True,
        )

        self.hidden2latent = nn.Linear(in_features=2 * self.lstm.hidden_size, out_features=256)

    def forward_one(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass with a single batch."""

        out, _ = self.lstm(batch.reshape(len(batch), 1, -1))
        out = self.hidden2latent(out[:, -1, :])

        return out

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        """SNN-style forward pass using cosine similarity."""

        out1 = self.forward_one(batch1)
        out2 = self.forward_one(batch2)

        return F.cosine_similarity(out1, out2, dim=-1)

    def run_train(self, logging: bool = False) -> None:
        """Runs the training loop."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.train()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3)

        for epoch in range(100):
            epoch_loss = 0
            for embs1, embs2, labels in self.dataloader:
                # Reshape and load onto the device
                embs1 = embs1.to(device)
                embs2 = embs2.to(device)
                labels = labels.float().to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                out = self(embs1, embs2).reshape(-1)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                # Increment the epoch loss for logging
                epoch_loss += loss.item()

            if logging and epoch % 10 == 9:
                print(f"Epoch: {epoch:2d} | Loss: {epoch_loss / len(self.dataloader):5f}")

            epoch_loss = 0


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

        # Further splits for training: required for the classification and evaluation algorithm
        df_train1 = df_train.sample(len(df_train) // 2, random_state=idx)
        df_train2 = df_train.drop(df_train1.index)

        # Generates embeddinngs for training
        df_train1["emb"] = df_train1["text"].apply(lambda s: model.encode(s))
        df_train1["lbl"] = df_train1["label"].apply(lambda l: ENCODE_PHENOTYPE[l])

        df_train2["emb"] = df_train2["text"].apply(lambda s: model.encode(s))
        df_test["emb"] = df_test["text"].apply(lambda s: model.encode(s))

        # Trains the SNN
        dataset = SNNTrainDataset(df_train1)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        snn = SiameseNeuralNetwork(train_dataloader)
        snn.run_train()

        # Generates embeddings using transformer models and convert to PyTorch tensors
        emb_train = snn.forward_one(torch.tensor(df_train2["emb"].array).to(device))
        emb_test = snn.forward_one(torch.tensor(df_test["emb"].array).to(device))

        # Encodes the labels and convert to PyTorch tensors
        lbl_train = torch.tensor(df_train2["label"].apply(lambda l: ENCODE_PHENOTYPE[l]).array)
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
