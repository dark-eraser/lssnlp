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
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from tqdm import tqdm

# Download necessary resources from NLTK
nltk.download('stopwords')


# %%
def preprocess_text(text):
    # Remove punctuation and special characters
    # print(text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text


# %%
#  Dump all artilces into one file as title category and text
# os.chdir("D:/nlp/")
# article_dump_folder = 'cleaned_articles'

# # List all JSON files in the folder
# json_files = [f for f in os.listdir(article_dump_folder) if f.endswith('.json')]

# # Extract text, title, and categories from each JSON file
# articles = []
# for json_file in json_files:
#     with open(os.path.join(article_dump_folder, json_file), 'r') as file:
#         article_data = json.load(file)
        
#         title = article_data['title']
#         text = article_data['text']
#         categories = article_data['categories']
        
#         articles.append({'title': title, 'text': text, 'categories': categories})

# %%
# with open("articles_dump.json", "w", encoding="utf8") as f:
#     json.dump(articles, f, indent=2, ensure_ascii=False)

# %%
os.chdir("D:/nlp/")
with open("articles_dump.json", "r", encoding="utf8") as f:
    articles = json.load(f)

# %%
# Preprocess the text
# preprocessed_texts = [preprocess_text(article['text'][0]) for article in tqdm(articles)]


# %%
# Create a TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)



# %%
# Get the feature names (words) from the vectorizer

# feature_names = vectorizer.get_feature_names_out()



# %%
# # Print the preprocessed text, title, and the corresponding TF-IDF matrix
# for article, preprocessed_text, tfidf_vector in (zip(articles, preprocessed_texts, tfidf_matrix)):
#     print('Title:', article['title'])
#     print('Preprocessed Text:', preprocessed_text)
#     print('TF-IDF Vector:', tfidf_vector.toarray())
#     print('------------------------------------')

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert the article titles into a list
all_articles = [article["title"] for article in articles]

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the article titles into a document-term matrix
dtm = vectorizer.fit_transform(all_articles)

# Compute the cosine similarity matrix




# %%
cosine_similarity_matrix = cosine_similarity(dtm)

# Iterate over the category pairs and calculate the cosine similarity
for i, category1 in tqdm(enumerate(categories)):
    for j, category2 in enumerate(categories):
        if i == j:
            similarity_matrix[i, j] = 1.0  # Similarity between the same category is 1.0
        elif i < j:
            category1_articles = category_articles[category1]
            category2_articles = category_articles[category2]
            
            # Get the indices of the articles in the document-term matrix
            category1_indices = [all_articles.index(article) for article in category1_articles]
            category2_indices = [all_articles.index(article) for article in category2_articles]
            
            # Extract the corresponding rows from the document-term matrix
            category1_dtm = dtm[category1_indices]
            category2_dtm = dtm[category2_indices]
            
            # Calculate the cosine similarity between the categories
            cosine_similarity_score = cosine_similarity(category1_dtm, category2_dtm)
            
            similarity_matrix[i, j] = cosine_similarity_score
            similarity_matrix[j, i] = cosine_similarity_score

# %%


# List of categories and their corresponding articles
categories = []
for article in tqdm(articles):
    categories.extend(article['categories'])
category_articles = {}






# %%
for article in tqdm(articles):
    for category in (article['categories']):
        if category not in category_articles:
            category_articles[category] = []
        category_articles[category].append(article["title"])

# %%
category_articles.__len__()

# %%
category_articles_filtered = {
    category: articles for category, articles in category_articles.items() if len(articles) >= 300}
category_articles_filtered.__len__()

categories = list(category_articles_filtered.keys())


# %%
categories.__len__()

# %%

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Convert the article titles into a list
# all_articles = [article["title"] for article in articles]

# # Initialize the TF-IDF vectorizer
# vectorizer = TfidfVectorizer()

# # Fit and transform the article titles into a document-term matrix
# dtm = vectorizer.fit_transform(all_articles)

# # Compute the cosine similarity matrix
# cosine_similarity_matrix = cosine_similarity(dtm)

# # Iterate over the category pairs and calculate the cosine similarity
# for i, category1 in tqdm(enumerate(categories)):
#     for j, category2 in enumerate(categories):
#         if i == j:
#             similarity_matrix[i, j] = 1.0  # Similarity between the same category is 1.0
#         elif i < j:
#             category1_articles = category_articles[category1]
#             category2_articles = category_articles[category2]
            
#             # Get the indices of the articles in the document-term matrix
#             category1_indices = [all_articles.index(article) for article in category1_articles]
#             category2_indices = [all_articles.index(article) for article in category2_articles]
            
#             # Extract the corresponding rows from the document-term matrix
#             category1_dtm = dtm[category1_indices]
#             category2_dtm = dtm[category2_indices]
            
#             # Calculate the cosine similarity between the categories
#             cosine_similarity_score = cosine_similarity(category1_dtm, category2_dtm)
            
#             similarity_matrix[i, j] = cosine_similarity_score
#             similarity_matrix[j, i] = cosine_similarity_score


# # Print the similarity matrix
# print(cosine_similarity_matrix)


# %%
from scipy.cluster.hierarchy import linkage, dendrogram

# Perform hierarchical clustering
linkage_matrix = linkage(similarity_matrix, method='complete')

# Generate a dendrogram for visualization
dendrogram(linkage_matrix, labels=categories)
