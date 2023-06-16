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
# %load_ext jupyternotify
import json
import nltk
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import concurrent.futures

import numpy as np

os.chdir("D:/nlp")

# %%
# Load the Wikipedia articles from the JSON file
with open('articles_dump.json', 'r') as file:
    wikipedia_articles = json.load(file)

# %%
# Preprocessing
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# %%
def preprocess_text(text):
    # Remove irrelevant information (links, citations, images) - You can use regex or specialized libraries for this
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Apply stemming or lemmatization
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)



# %%
# Process and preprocess the Wikipedia articles
preprocessed_articles = []
for article in tqdm(wikipedia_articles):
    title = article['title']
    texts = article['text']
    categories = article['categories']
    
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    preprocessed_articles.append({
        'title': title,
        'preprocessed_texts': preprocessed_texts,
        'categories': categories
    })

# %%



# %%
# List of clusters containing Wikipedia categories
clusters = []
with open('1000clusters_10000comp.json', 'r') as file:
    for line in file:
        clusters.append(line.strip().split(', '))

# %%
# Select a transformer model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# %%
# Generate embeddings for preprocessed articles
embeddings = []
for article in tqdm(preprocessed_articles):
    input_ids = tokenizer.encode(article['preprocessed_text'], add_special_tokens=True)
    inputs = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(inputs)
        article_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        embeddings.append(article_embedding)
