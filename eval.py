import torch
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, Trainer
from utils import load_and_prepare
from datasets import Dataset
from sklearn.metrics import classification_report

def main():
    # 載入資料
    df = load_and_prepare("data_final/CICIDS2017.csv")
    dataset = Dataset.from_pandas(df[["text", "label"]])
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # 載入模型 & tokenizer
    model = ElectraForSequenceClassification.from_pretrained("models/electra-ids")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize_fn, batched=True)

    trainer = Trainer(model=model, tokenizer=tokenizer)

    preds = trainer.predict(dataset["test"])
    y_true = dataset["test"]["label"]
    y_pred = preds.predictions.argmax(axis=1)

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
