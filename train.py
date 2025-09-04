import os
import torch
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from utils import load_and_prepare

def main():
    # === Step 1. 載入資料 ===
    df = load_and_prepare("data_final/CICIDS2017.csv")

    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # === Step 2. Tokenizer ===
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_fn, batched=True)

    # === Step 3. 建立模型 ===
    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-small-discriminator",
        num_labels=dataset["train"].features["label"].num_classes
    )

    # === Step 4. 訓練參數 ===
    training_args = TrainingArguments(
        output_dir="models",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="results/logs",
        logging_steps=50,
    )

    # === Step 5. Trainer ===
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )

    # === Step 6. 開始訓練 ===
    trainer.train()
    trainer.save_model("models/electra-ids")

if __name__ == "__main__":
    main()
