import json
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger


class ArticleDataset(object):
    def __init__(self, articles_dir, clustering_fpath, tokenizer):
        self.articles_dir = articles_dir
        self.articles_file_paths = [os.path.join(articles_dir, fname) for fname in os.listdir(articles_dir)]

        # Assuming clustering_fpath CSV has two columns: 'category' and 'label'
        self.clusters_mapping = pd.read_csv(clustering_fpath).set_index("key").to_dict()["cluster_label"]

        self.num_classes = len(set(self.clusters_mapping.values())) - 1  # we don't count the noise as a label

        self.tokenizer = tokenizer
        logger.info("Finished dataset init")

    def __len__(self):
        return len(self.articles_file_paths)

    # iter method to get each element at the time and tokenize it using bert
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

        # encode the sequence and add padding
        inputs = self.tokenizer(
            full_text,
            # add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "labels": torch.tensor(encoded_cats, dtype=torch.float),
        }

    def __len__(self):
        return len(self.articles_file_paths)
