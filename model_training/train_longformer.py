import torch
from datasets import ArticleDataset
from loguru import logger
from metrics import compute_metrics

# from optimizers import get_optimizers
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LongformerModel, Trainer, TrainingArguments
from transformers.models.longformer.modeling_longformer import (
    LongformerClassificationHead,
    LongformerPreTrainedModel,
)

WINDOW_SIZE = 1024


# instantiate a Longformer for multilabel classification class
class LongformerForMultiLabelSequenceClassification(LongformerPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task.
    This instance takes the pooled output of the LongFormer based model and spasses it through a classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities for all the labels that we feed into the model.
    """

    def __init__(self, config):
        super(LongformerForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models. This is taken care of by HuggingFace
        # on the LongformerForSequenceClassification class
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        # pass arguments to longformer model
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size).
        sequence_output = outputs["last_hidden_state"]

        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs


clustering_fpath = "data/clusters/hdbscan_results_20_200.csv"
articles_dir_train = "/home/kajetan_pyszkowski/articles/train"
articles_dir_test = "/home/kajetan_pyszkowski/articles/validate"

# create a class to process the training and test data
tokenizer = AutoTokenizer.from_pretrained(
    "allenai/longformer-base-4096", padding="max_length", truncation=True, max_length=WINDOW_SIZE
)

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

model = LongformerForMultiLabelSequenceClassification.from_pretrained(
    "allenai/longformer-base-4096",
    gradient_checkpointing=True,
    attention_window=WINDOW_SIZE,
    num_labels=training_data.num_classes,
    # cache_dir="/home/kajetan_pyszkowski/cache_dir",
    return_dict=True,
)

training_args = TrainingArguments(
    output_dir="outputs",
    # logging_dir="logs",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
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
    run_name=f"longformer_{WINDOW_SIZE}",
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
