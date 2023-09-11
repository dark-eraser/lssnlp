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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import json
from sklearn.feature_extraction.text import HashingVectorizer

os.chdir("D:/nlp")

# %%
# Assuming you have your data stored in a variable called 'data'
with open('categories.json', 'r', encoding='utf-8') as file:
    categories = json.load(file)

# %%
# flatten categories with set comprehension 

categories = {item for sublist in categories for item in sublist}



# %%
categories=list(categories)

# %%
categories.__len__()

# %%
# Set the batch size for vectorization and PCA
from sklearn.decomposition import IncrementalPCA

batch_size = 1000

# Vectorize the categories in batches using HashingVectorizer
vectorizer = HashingVectorizer()
num_samples = len(flattened_categories)
num_batches = int(np.ceil(num_samples / batch_size))
pca = IncrementalPCA(n_components=100)  # Adjust the number of components as needed
scaled_vectors = None

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)
    batch_categories = flattened_categories[start_idx:end_idx]
    batch_vectors = vectorizer.transform(batch_categories)
    batch_vectors_array = batch_vectors.toarray()

    if scaled_vectors is None:
        scaled_vectors = batch_vectors_array
    else:
        scaled_vectors = np.vstack((scaled_vectors, batch_vectors_array))

    # Apply incremental PCA on the accumulated vectors
    if i > 0 and i % 10 == 0:  # Adjust the frequency of PCA updates as needed
        scaled_vectors = pca.partial_fit(scaled_vectors)

# %%
# Convert the sparse matrix to an array
scaled_vectors = pca.transform(scaled_vectors)


# %%
# Create a DBSCAN object with your desired parameters
eps = 0.5  # The maximum distance between two samples to be considered as neighbors
min_samples = 2  # The minimum number of samples in a neighborhood to be considered as a core point
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# %%
# Fit the DBSCAN model to the scaled vectors
dbscan.fit(scaled_vectors)

# %%
# Retrieve the predicted labels and core sample indices
labels = dbscan.labels_
core_samples = dbscan.core_sample_indices_

# %%
# Get the number of clusters found by DBSCAN (-1 represents noise/outliers)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:", num_clusters)

# %%
# Iterate over each cluster and print the categories
for cluster_id in set(labels):
    if cluster_id == -1:
        # Skip noise/outliers
        continue
    cluster_categories = [categories[i] for i in range(len(categories)) if labels[i] == cluster_id]
    print("Cluster ID:", cluster_id)
    print("Categories:", cluster_categories)

# %%
# Access the core sample indices
print("Core Sample Indices:", core_samples)
