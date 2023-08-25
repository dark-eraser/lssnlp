import logging
import pickle

import hdbscan
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MEMORY_LOC = "cache"
CORE_DIST_N_JOBS = 32

with open("categories.pickle", "rb") as f:
    counter = pickle.load(f)
keys = list(counter.keys())

embeddings = np.load("embeddings.npy", mmap_mode="r+")

logger.info(f"Loaded embeddings with shape {embeddings.shape}")

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, memory=MEMORY_LOC, core_dist_n_jobs=CORE_DIST_N_JOBS)
cluster_labels = clusterer.fit_predict(embeddings)

logger.info(f"Done with HDBSCAN clustering")

# Save the results to a CSV file
results = pd.DataFrame({"key": keys, "cluster_label": cluster_labels})
results.to_csv("hdbscan_results.csv", index=False)

logger.info(f"Results saved to 'hdbscan_results.csv'")
