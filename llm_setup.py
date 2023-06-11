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
import json
import glob
import os
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
os.chdir("D:/nlp")
# List all JSON files in a directory
json_files = glob.glob("cleaned_articles/*.json")

# %%
# Load and preprocess the JSON data
text_data = ""
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
        for item in data:
            text_data += item["text"] + "\n"

# %%
# Initialize the tokenizer and model
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

# %%
# Tokenize the text data
tokenized_data = tokenizer.tokenize(text_data)

# %%
# Convert the tokenized data to input sequences
input_sequences = tokenizer.encode(tokenized_data, add_special_tokens=False)

# %%
# Create a TextDataset for training
dataset = TextDataset(
    tokenizer=tokenizer,
    tokenized_text=input_sequences,
    block_size=128  # Adjust the block size according to your data and memory constraints
)

# %%
# Create a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Set to True if you have masked language modeling (MLM) training data
)

# %%
# Define the training arguments
training_args = TrainingArguments(
    output_dir="model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# %%
# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# %%
# Start the training
trainer.train()

# %%
# Save the trained model
trainer.save_model("trained_model")
