import json
import os
import pickle
from collections import Counter

ARTICLE_DIR = "data/articles/"


files = os.listdir(ARTICLE_DIR)

categories = Counter()

for file in files:
    with open(ARTICLE_DIR + file, "r") as f:
        data = json.load(f)

    cat = data["categories"]

    for c in cat:
        categories.update([c])

with open("data/categories/categories.pickle", "wb") as f:
    pickle.dump(categories, f)
