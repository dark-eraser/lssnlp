import logging
import pickle

import cupy as cp
import numpy as np
import pandas as pd
from cuml.cluster import HDBSCAN
from cuml.common.device_selection import set_global_device_type
from cuml.manifold import UMAP
from cuml.preprocessing import normalize

set_global_device_type("gpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


umap_n_neighbors = 100
umap_min_dist = 1e-3
umap_spread = 2.0
umap_n_epochs = 500
umap_random_state = 42


with open("categories.pickle", "rb") as f:
    counter = pickle.load(f)
keys = list(counter.keys())

embeddings = np.load("embeddings.npy", mmap_mode="r+")
logger.info(f"Loaded embeddings with shape {embeddings.shape}")

X = normalize(cp.asarray(embeddings))
logger.info(f"Normalized embeddings")


umap = UMAP(
    n_neighbors=umap_n_neighbors,
    min_dist=umap_min_dist,
    spread=umap_spread,
    n_epochs=umap_n_epochs,
    random_state=umap_random_state,
)
reduced_embeddings = umap.fit_transform(X)
logger.info(f"Done with UMAP")

cp.save("reduced_embeddings.npy", reduced_embeddings)
logger.info(f"Saved reduced embeddings to 'reduced_embeddings.npy'")

# Run HDBSCAN
clusterer = HDBSCAN(min_cluster_size=1000, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(reduced_embeddings)

logger.info(f"Done with HDBSCAN clustering")

# Save the results to a CSV file
results = pd.DataFrame({"key": keys, "cluster_label": cluster_labels})
results.to_csv("hdbscan_results.csv", index=False)

logger.info(f"Results saved to 'hdbscan_results.csv'")
