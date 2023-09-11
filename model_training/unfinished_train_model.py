# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: nlplss
#     language: python
#     name: python3
# ---

# %%
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import os
import numpy as np

os.chdir("D:/nlp")

# %%
# Load the Wikipedia articles from the JSON file
with open('preprocessed_text.json.json', 'r', encoding='utf-8') as file:
    preprocessed_articles = json.load(file)

# %%
# List of clusters containing Wikipedia categories
clusters = []
with open('1000clusters_10000comp.json',encoding="utf-8") as file:
    for line in file:
        clusters.append(line.strip().split(', '))

# %%
# Select a transformer model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# %%
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
import multiprocessing
from tqdm import tqdm

# Function to process a single article
def process_article(article):
    # Move the model to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("device:",device)
    preprocessed_texts = article['preprocessed_texts']

    # Tokenize the preprocessed texts
    tokenized_texts = [word_tokenize(text) for text in preprocessed_texts]

    # Convert tokenized texts to input IDs
    input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_texts]

    # Pad sequences to the same length
    max_length = max(len(ids) for ids in input_ids)
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]

    # Convert input IDs to tensors and move to the GPU
    input_ids_tensor = torch.tensor(padded_input_ids).to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_ids_tensor)
        article_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return article_embedding

# Set the number of worker processes for parallel processing
num_workers = multiprocessing.cpu_count()

# Create a DataLoader to batch the processing
dataset = preprocessed_articles
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=lambda batch: batch)

# Generate embeddings for preprocessed articles
embeddings = []

def worker_fn(batch):
    batch_embeddings = []

    # Process the batch
    for article in tqdm(batch, desc='Processing articles'):
        article_embedding = process_article(article)
        batch_embeddings.append(article_embedding)

    return batch_embeddings

# Spawn worker processes and process the batches in parallel
with torch.multiprocessing.spawn(worker_fn, args=(dataloader,), nprocs=num_workers) as processes:
    for batch_embeddings in processes:
        embeddings.extend(batch_embeddings)

    
