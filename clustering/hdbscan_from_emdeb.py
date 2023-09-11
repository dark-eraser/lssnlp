import logging
import pickle
import time

import cupy as cp
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.common.device_selection import set_global_device_type
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

set_global_device_type("gpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


umap_n_neighbors = 100
umap_min_dist = 1e-3
umap_spread = 2.0
umap_n_epochs = 500
umap_random_state = 42


with open("data/categories/categories.pickle", "rb") as f:
    counter = pickle.load(f)
keys = list(counter.keys())

# embeddings = np.load("embeddings.npy", mmap_mode="r+")
# logger.info(f"Loaded embeddings with shape {embeddings.shape}")

# X = normalize(cp.asarray(embeddings))
# logger.info(f"Normalized embeddings")


# umap = UMAP(
#     n_components=2,
#     n_neighbors=umap_n_neighbors,
#     min_dist=umap_min_dist,
#     spread=umap_spread,
#     n_epochs=umap_n_epochs,
#     random_state=umap_random_state,
#     verbose=5,
# )
# reduced_embeddings = umap.fit_transform(X)
# logger.info(f"Done with UMAP")

# cp.save("reduced_embeddings_2.npy", reduced_embeddings)
# logger.info(f"Saved reduced embeddings to 'reduced_embeddings.npy'")


reduced_embeddings = cp.load("reduced_embeddings_2.npy")
logger.info(f"Shape : {reduced_embeddings.shape}")
# Run HDBSCAN
now = time.time()
clusterer = HDBSCAN(min_cluster_size=200, gen_min_span_tree=True, verbose=5)
cluster_labels = clusterer.fit_predict(reduced_embeddings)
logger.info(f"Done with HDBSCAN clustering in {time.time() - now} seconds")

percentage = len(cluster_labels[cluster_labels != -1]) / reduced_embeddings.shape[0]
logger.info(f"Meaningfull percentage: {round(percentage,2)}")

# Save the results to a CSV file
results = pd.DataFrame({"key": keys, "cluster_label": cluster_labels.get()})
results.to_csv("hdbscan_results.csv", index=False)

logger.info(f"Results saved to 'hdbscan_results.csv'")

x = reduced_embeddings[:, 0].get()
y = reduced_embeddings[:, 1].get()

x = x[cluster_labels > -1].get()
y = y[cluster_labels > -1].get()
labels_no_noise = cluster_labels[cluster_labels > -1].get()

x_noise = reduced_embeddings[:, 0]
y_noise = reduced_embeddings[:, 1]

x_noise = x_noise[cluster_labels == -1]
y_noise = y_noise[cluster_labels == -1]

figure(figsize=(10, 7), dpi=80)

plt.scatter(x, y, c=labels_no_noise, s=0.1, cmap="Spectral")
