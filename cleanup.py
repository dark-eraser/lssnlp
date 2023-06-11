from extract import Extractor
import json
import re
import os
from tqdm import tqdm

CLEANED_ARTICLES_DIR = "cleaned_articles"
PARSED_ARTICLES_DIR = "parsed_articles"
def main():


    os.chdir("D:/nlp")
    os.makedirs(CLEANED_ARTICLES_DIR, exist_ok=True)
    all_files = os.listdir(PARSED_ARTICLES_DIR)

    # Create an extractor object
    extractor = Extractor()
    for file in tqdm(all_files):

        with open(os.path.join(PARSED_ARTICLES_DIR, file), "r") as f:
            doc = json.load(f)

        # Extract the data
        paragraph_list = extractor.clean_text(doc[1], mark_headers=True, html_safe=True)
        # print(paragraph_list)
        if len(paragraph_list) == 0:
            continue
        paragraph_list[0]= re.sub(r'\([^)]*\)', '', paragraph_list[0]).replace("  ", " ")

        cleaned_article = {
            "title": doc[0],
            "text": paragraph_list,
            "categories": doc[2],
        }

        with open(os.path.join(CLEANED_ARTICLES_DIR, file), "w", encoding="utf-8") as f:
            json.dump(cleaned_article, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    main()