import torch
from datasets import ArticleDataset
from loguru import logger
from metrics import compute_metrics

# from optimizers import get_optimizers
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

WINDOW_SIZE = 1024

clustering_fpath = "data/clusters/hdbscan_results_20_200.csv"
articles_dir_train = "/home/kajetan_pyszkowski/articles/train"
articles_dir_test = "/home/kajetan_pyszkowski/articles/validate"

# create a class to process the training and test data
# tokenizer = AutoTokenizer.from_pretrained(
#     "gpt2", padding="max_length", truncation=True, max_length=WINDOW_SIZE
# )
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

training_data = ArticleDataset(articles_dir_train, clustering_fpath, tokenizer)
test_data = ArticleDataset(articles_dir_test, clustering_fpath, tokenizer)

# use the dataloaders class to load the data
dataloaders_dict = {
    "train": DataLoader(
        training_data,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ),
    "test": DataLoader(test_data, shuffle=True, num_workers=4, pin_memory=True),
}

dataset_sizes = {"train": len(training_data), "test": len(test_data)}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2",
    # gradient_checkpointing=True,
    num_labels=training_data.num_classes,
    # cache_dir="/home/kajetan_pyszkowski/cache_dir",
    return_dict=True,
    problem_type="multi_label_classification",
)

training_args = TrainingArguments(
    output_dir="outputs",
    # logging_dir="logs",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    # gradient_accumulation_steps=2,
    # gradient_checkpointing=False,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    disable_tqdm=False,
    load_best_model_at_end=True,
    warmup_steps=1500,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=8,
    fp16=True,
    # tf32=True,
    optim="adafactor",
    run_name=f"gpt2_{WINDOW_SIZE}",
    # torch_compile=True,
)

# optimizer = get_optimizers(model, training_args)
# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    # optimizers=(optimizer, None),
)

trainer.train()
