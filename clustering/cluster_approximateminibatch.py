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
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
os.chdir("D:/nlp")


# %%
# Load the similarity matrix from the file
similarity_matrix = np.load('similarity_matrix.npy')

# %%
# Convert the similarity matrix to float32 data type
similarity_matrix = similarity_matrix.astype(np.float32)

# %%
import json
with open("articles_dump.json", "r", encoding="utf8") as f:
    articles = json.load(f)

# %%
# List of categories and their corresponding articles
categories = []
category_articles = {}

for article in tqdm(articles):
    article_categories = article['categories']
    categories.extend(article_categories)  # Append the list of categories

    for category in (article_categories):
        category_name = category.split(":")[1]
        if category_name not in category_articles:
            category_articles[category_name] = []
        category_articles[category_name].append(article["title"])


# %%
# for article in tqdm(articles):
#     for category in (article['categories']):
#         if category.split(":")[1] not in category_articles:
#             category_articles[category.split(":")[1]] = []
#         category_articles[category.split(":")[1]].append(article["title"])

# %%
category_articles_filtered = {
    category: articles for category, articles in category_articles.items() if len(articles) >= 10}
category_articles_filtered.__len__()

categories = list(category_articles_filtered.keys())

# %%
# Reduce the dimensionality of the similarity matrix using TruncatedSVD
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from tqdm import tqdm
n_components = 10000  # Adjust the number of components as needed
svd = TruncatedSVD(n_components=n_components)
similarity_matrix_reduced = svd.fit_transform(similarity_matrix)


# %%
# Perform clustering using KMeans on the reduced matrix
n_clusters = 1000  # Adjust the number of clusters as needed
clusterer = KMeans(n_clusters=n_clusters)

# %%
# Get the cluster labels
labels = clusterer.fit_predict(similarity_matrix_reduced)

# Print the clusters
clusters = [[] for _ in range(n_clusters)]
for i, category in tqdm(enumerate(categories)):
    clusters[labels[i]].append(category)

# %%


for cluster_idx, cluster in tqdm(enumerate(clusters)):
    print(f"Cluster {cluster_idx}: {cluster}")

with open("1000clusters_10000comp.json", "w", encoding="utf8") as f:
    json.dump(clusters, f, indent=2, ensure_ascii=False)

# %%
