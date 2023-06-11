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
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import os
from tqdm import tqdm
os.chdir("D:/nlp")
import json

# %%
categories=[]
for files in tqdm(os.listdir("cleaned_articles")):
    with open(os.path.join("cleaned_articles", files), "r", encoding="utf8") as f:
        data = json.load(f)
        if data["categories"] not in categories:
            categories.append(data["categories"])


# %%


# %%
with open("categories.json", "w",encoding="utf8") as f:
    json.dump(categories, f, indent=2, ensure_ascii=False)

# %%
