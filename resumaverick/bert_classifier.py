from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
import numpy as np

from utils import load_resume_dataset

from datasets import Dataset
from transformers import Trainer, TrainingArguments
from evaluate import load as load_metric
from sklearn.model_selection import train_test_split
from pathlib import Path
from augmentation import synonym_replace
from tqdm import tqdm

tqdm.pandas()


def load_bert_model(model_name: str):
    """
    Load the BERT model from the given path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_bert_model("distilbert-base-uncased")


def preprocess_function(examples: dict[str, list[str]]) -> dict[str, list[int]]:
    """
    Tokenize input texts under key 'text' in the examples batch.
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def prepare_data(
                        train_X: pd.Series,
                        train_y: pd.Series,
                        val_X: pd.Series,
                        val_y: pd.Series) -> tuple[Dataset, Dataset]:
    """
    Finetune the BERT model on the given training and validation data.
    """
    train_dataset = Dataset.from_dict({"text": train_X, "label": train_y})
    eval_dataset = Dataset.from_dict({"text": val_X, "label": val_y})
    print(f"Train dataset: {train_dataset.features}")
    print(f"Eval dataset: {eval_dataset.features}")
    print(f"Train dataset: {train_dataset.shape}")
    print(f"Eval dataset: {eval_dataset.shape}")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    return train_dataset, eval_dataset


accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted") # or "binary" if 2 classes
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }


def finetune_bert_model(model: AutoModelForSequenceClassification,
                        train_dataset: Dataset,
                        eval_dataset: Dataset,
                        ) -> AutoModelForSequenceClassification:
    """
    Finetune the BERT model on the given training and validation data.
    """

    training_args = TrainingArguments(
                        output_dir="./results",
                        optim="adamw_torch",
                        learning_rate=1e-5,
                        num_train_epochs=20,
                        per_device_train_batch_size=32,
                        per_device_eval_batch_size=8,
                        warmup_steps=10,
                        weight_decay=0.01,
                        logging_dir='./logs',
                        eval_strategy="steps",
                        eval_steps=50,
                        logging_steps=50, # for training loss
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy",
                        # report locally:
                        report_to="none",
                        run_name="bart-classifier-run",
                    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    Path("./models").mkdir(parents=True, exist_ok=True)
    trainer.save_model("./models/bert-classifier")
    return model


if __name__ == "__main__":
    resume_csv_path = Path(__file__).parent.parent / "data" / "Resume.csv"
    resume_df = pd.read_csv(resume_csv_path)
    # Text inputs and original string labels
    resume_df_X = resume_df["Resume_str"]
    resume_df_y_str = resume_df["Category"]
    print(f'applying synonym augmentation to {len(resume_df_X)} resumes')
    augmented_df_X = resume_df_X.progress_apply(synonym_replace)
    resume_df_X = pd.concat([resume_df_X, augmented_df_X])
    resume_df_y_str = pd.concat([resume_df_y_str, resume_df_y_str])
    # Map string labels to integer IDs
    classes = sorted(resume_df_y_str.unique())
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for label, idx in label2id.items()}
    resume_df_y = resume_df_y_str.map(label2id)
    num_labels = len(classes)
    # Update the model's classification head and config to match the label count
    hidden_size = getattr(model.config, "hidden_size", model.config.dim)

    # Sync config and model attributes for number of labels
    model.config.num_labels = num_labels
    model.num_labels = num_labels
    mlp = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, num_labels),
    )
    print(f"Classifier before: {model.classifier}, weight shape: {model.classifier.weight.shape}")
    model.classifier = mlp
    print(f"Classifier after: {model.classifier}")
    model.config.id2label = id2label
    model.config.label2id = label2id
    # Reset problem type to ensure correct loss computation
    model.config.problem_type = None
    X_Train_Val, X_Test, y_Train_Val, y_Test = train_test_split(resume_df_X, resume_df_y, test_size=0.2, stratify=resume_df_y, random_state=42)
    X_Train, X_Val, y_Train, y_Val = train_test_split(X_Train_Val, y_Train_Val, test_size=0.2, stratify=y_Train_Val, random_state=42)

    train_dataset, eval_dataset = prepare_data(X_Train, y_Train, X_Val, y_Val)
    finetuned_model = finetune_bert_model(model, train_dataset, eval_dataset)
    # Prepare the test dataset
    test_dataset = Dataset.from_dict({"text": X_Test, "label": y_Test})
    test_dataset = test_dataset.map(preprocess_function, batched=True)



    
