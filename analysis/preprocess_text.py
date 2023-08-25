import json
import nltk
import concurrent.futures
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import os

# Preprocessing
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

os.chdir("D:/nlp")
def preprocess_text(text):
    # Remove irrelevant information (links, citations, images) - You can use regex or specialized libraries for this
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Apply stemming or lemmatization
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

def process_chunk(chunk):
    preprocessed_articles_chunk = []

    for article in tqdm(chunk, desc='Processing chunk '):
        title = article['title']
        categories = article['categories']
        texts = article['text']
        # Process and preprocess the Wikipedia articles
        preprocessed_texts = [preprocess_text(text) for text in texts]
        preprocessed_articles_chunk.append({
            'title': title,
            'preprocessed_texts': preprocessed_texts,
            'categories': categories
        })
    return preprocessed_articles_chunk

def process_json_file(file_path, chunk_size=1000, num_threads=16):
    with open(file_path, 'r',encoding="utf-8") as file:
        data = json.load(file)

    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    preprocessed_articles = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(process_chunk, chunks)
        for chunk_results in results:
            preprocessed_articles.extend(chunk_results)

    

    with open('preprocessed_text.json'+file_path, 'w') as file:
        json.dump(preprocessed_articles, file)
# Usage
process_json_file('articles_dump.json', num_threads=16)

