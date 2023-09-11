import json
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# Load and preprocess the data
# data = pd.read_csv("../data/clusters/hdbscan_results_20_200.csv")
# data["key"] = data["key"].str.replace("Category:", "")
# data = data[data["cluster_label"] != -1]  # Remove noise cluster


class ArticlesDataset(Dataset):
    def __init__(self, articles_dir, clustering_fpath, tokenizer):
        self.articles_dir = articles_dir
        self.articles_file_paths = [os.path.join(articles_dir, fname) for fname in os.listdir(articles_dir)]

        # Assuming clustering_fpath CSV has two columns: 'category' and 'label'
        self.clusters_mapping = pd.read_csv(clustering_fpath).set_index("key").to_dict()["cluster_label"]

        self.num_classes = len(set(self.clusters_mapping.values())) - 1  # we don't count the noise as a label

        self.tokenizer = tokenizer
        logger.info("Finished init")

    def __len__(self):
        return len(self.articles_file_paths)

    def __getitem__(self, idx):
        with open(self.articles_file_paths[idx], "r") as f:
            article = json.load(f)

        title = article["title"]
        text = article["text"]
        full_text = f"Title: {title} \n {text}"
        categories = article["categories"]

        encoded_cats = np.zeros(self.num_classes, dtype=int)

        for cat in categories:
            cat_num = self.clusters_mapping.get(cat, -1)
            if cat_num != -1:
                encoded_cats[cat_num] = 1

        inputs = self.tokenizer(
            full_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(encoded_cats, dtype=torch.float),
        }


# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"

# Assuming clustering_fpath and data directories are defined
clustering_fpath = "data/clusters/hdbscan_results_20_200.csv"
articles_dir_train = "/home/kajetan_pyszkowski/articles/train"
articles_dir_test = "/home/kajetan_pyszkowski/articles/validate"

# Create Dataset
tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset = ArticlesDataset(articles_dir_train, clustering_fpath, tokenizer=tokenizer)
test_dataset = ArticlesDataset(articles_dir_test, clustering_fpath, tokenizer=tokenizer)

logger.info("Initialied datasets")

# Load BERT model for sequence classification
num_classes = train_dataset.num_classes
logger.info(f"Setting up BERT with {num_classes} outputs")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Fine-tuning settings
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


# Fine-tuning loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device {device}")
model.to(device)

for epoch in range(5):  # You can adjust the number of epochs
    model.train()
    total_loss = 0.0  # To accumulate the total loss for the epoch
    for batch_num, batch in tqdm(enumerate(train_loader)):
        batch = tuple(batch[t].to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print progress at certain intervals (e.g., every 10 batches)
        if (batch_num + 1) % 1000 == 0:
            avg_batch_loss = total_loss / (batch_num + 1)
            logger.info(
                f"Epoch [{epoch+1}/{5}], Batch [{batch_num+1}/{len(train_loader)}], Avg. Loss: {avg_batch_loss:.4f}"
            )

    avg_epoch_loss = total_loss / len(train_loader)
    logger.info(f"Epoch [{epoch+1}/{5}], Avg. Loss: {avg_epoch_loss:.4f}")

    # Save the model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # model.save_pretrained(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}.pth"))

    logger.info("Evaluating model")
    # Evaluate the model on the test set
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for batch_num, batch in tqdm(enumerate(test_loader)):
        batch = tuple(batch[t].to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        predictions = (probs > 0.5).float()

        all_labels.append(inputs["labels"].cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

        # total_correct += (logits.argmax(dim=-1) == batch[2]).sum().item()
        # total_samples += len(batch[2])

    avg_epoch_loss = total_loss / len(test_loader)

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)

    f1 = f1_score(all_labels, all_predictions, average="micro")
    accuracy = accuracy_score(all_labels, all_predictions)

    logger.info(f"Avg. Loss: {avg_epoch_loss:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
