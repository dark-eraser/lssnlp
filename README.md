# lssnlp

Welcome to our WikiTopic project.

## Usage

### Download and parse the dataset

First download a wikipedia dump from wikimedia, it can take up to 12 hours as the file is at least 20Gb and the download speed is limited to.

https://wikidata.aerotechnet.com/enwiki/20230901/enwiki-20230901-pages-articles.xml.bz2

Then install the parserfromhell utility and use it to extract the articles. It should generate a folder with files called
"parsed*articles*{i}.json".

Then extract all the unique categories using the script `categories.py`

### Reduce the dataset

First you have to gen. the embeddings, use the `cat_to_bert_emb.py` script. It will generate a embeddings.npy file containing the non reduced embeddings with a shape of (#articles that you managed to extract, 762). Use a GPU setup as this can take some time.

Then reduce the dimentionality and cluster them using the hdbscan_from_embed.py script. It will generate a file called "clusters.npy" containing the cluster labels for each article. However keep in mind that we use cuML librairies and the install procedure can be a bit tricky.

### Then split the dataset into train and test

Use the script the `split_data.py` to split the dataset into two subfolders.

### Train a model

Then you have access to all our training procedures in the model_training folder. Use any of the files with the correct path (that are usally defined on the top of the file) to run the training procedure. It should automatically save the model in the model_output folder.
Also keep in mind that we used a GPU setup with a lot of vRAM, your training might fail if you don't have enough vRAM.

## Questions?

We stay available to help you reproduce the results, as we know that the procedure is quite long and tricky. Feel free to contact us at any time.
