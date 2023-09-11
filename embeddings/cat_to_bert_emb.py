import logging
import pickle
import time
from collections import Counter

import hdbscan
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BATCH_SIZE = 1000
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Let's assume you have a counter object named 'counter'.
with open("data/categories/categories.pickle", "rb") as f:
    counter = pickle.load(f)
# counter = Counter({"string1": 1, "string2": 2, "string3": 3, "string4": 4})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

model = model.to("cuda")

keys = list(counter.keys())[:10]
embeddings = []


# Wrap the range object with tqdm for a progress bar
for i in tqdm(range(0, len(keys), BATCH_SIZE), desc="Generating embeddings"):
    batch = keys[i : i + BATCH_SIZE]

    # Convert the keys to embeddings
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    embeddings.extend(batch_embeddings)
    if i % 10_000 == 0:
        logger.info(f"Emptying CUDA cache")
        torch.cuda.empty_cache()

# Convert list of embeddings to numpy array for HDBSCAN
embeddings = np.vstack(embeddings)

for embedding in embeddings:
    print(embedding[0:10])

logger.info(f"Done generating embeddings with shape {embeddings.shape}")
# Save the embeddings to a file
np.save("embeddings.npy", embeddings)
