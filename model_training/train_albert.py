import torch
from datasets import ArticleDataset
from metrics import compute_metrics
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

WINDOW_SIZE = 1024
MODEL_NAME = "albert-base-v2"

clustering_fpath = "data/clusters/hdbscan_results_20_200.csv"
articles_dir_train = "/home/kajetan_pyszkowski/articles/train"
articles_dir_test = "/home/kajetan_pyszkowski/articles/validate"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

model = AlbertForSequenceClassification.from_pretrained(
    MODEL_NAME,
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
    per_device_train_batch_size=22,
    gradient_accumulation_steps=2,
    # gradient_checkpointing=False,
    per_device_eval_batch_size=64,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=4890,
    save_steps=4890,
    logging_steps=200,
    disable_tqdm=False,
    load_best_model_at_end=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    optim="adafactor",
    run_name=f"albert",
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
