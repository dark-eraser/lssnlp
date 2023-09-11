import os
import json
import random
import shutil

# Set the paths for the source folder and the destination folders
source_folder = '../data/articles/cleaned_articles'
train_folder = '../data/articles/train'
val_folder = '../data/articles/validate'

# Load the list of article filenames
article_filenames = [filename for filename in os.listdir(source_folder) if filename.endswith('.json')]

# Shuffle the list of filenames randomly
random.shuffle(article_filenames)

# Calculate the number of articles for each split
total_articles = len(article_filenames)
train_size = int(0.8 * total_articles)
val_size = total_articles - train_size

# Create the train and validation folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Move articles to the train and validation folders
for i, filename in enumerate(article_filenames):
    source_path = os.path.join(source_folder, filename)
    if i < train_size:
        destination_path = os.path.join(train_folder, filename)
    else:
        destination_path = os.path.join(val_folder, filename)
    shutil.copyfile(source_path, destination_path)

print("Splitting and randomizing complete.")
