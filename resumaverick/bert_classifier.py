from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
import numpy as np

from datasets import Dataset
from transformers import Trainer, TrainingArguments
from evaluate import load as load_metric
from sklearn.model_selection import train_test_split
from pathlib import Path
from augmentation import apply_multiple_augmentations, synonym_replace, back_translate, shuffle_summary
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback


tqdm.pandas()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PrintLRCallback(TrainerCallback):
    def on_step_end(self, args, state, _control, **kwargs):
        optimizer = kwargs['optimizer']
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {state.global_step}: LR={current_lr:.8f}")



def load_bert_model(model_name: str):
    """
    Load the BERT model from the given path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


model_name = "distilbert-base-uncased" if device.type == "cuda" else "distilbert-base-uncased"
print(f'running on {device.type} choosing model: {model_name}')
tokenizer, model = load_bert_model(model_name)


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
    # Move model to the target device (GPU if available)
    model.to(device)
    num_epochs = 24
    batch_size = 32
    steps_per_epoch = len(train_dataset) // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=num_epochs * steps_per_epoch
                                                        )
    fp16 = device.type == "cuda" # True if GPU, False if CPU
    training_args = TrainingArguments(
                        output_dir="./results",
                        fp16=fp16, # mixed precision, speedup on T4s and A100s
                        num_train_epochs=num_epochs,
                        per_device_train_batch_size=batch_size,
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
        optimizers=(optimizer, scheduler),
        callbacks=[PrintLRCallback()],
    )
    trainer.train()
    
    Path("./models").mkdir(parents=True, exist_ok=True)
    trainer.save_model("./models/bert-classifier")
    return model


if __name__ == "__main__":
    resume_csv_path = Path(__file__).parent.parent / "data" / "Resume.csv"
    resume_df = pd.read_csv(resume_csv_path)
    # Map string labels to integer IDs
    classes = sorted(resume_df["Category"].unique())
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for label, idx in label2id.items()}
    resume_df_y = resume_df["Category"].map(label2id)
    resume_df_X = resume_df["Resume_str"]
    
    num_labels = len(classes)
    # Update the model's classification head and config to match the label count
    hidden_size = model.config.hidden_size

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
    # model.classifier = mlp
    model.classifier = torch.nn.Linear(hidden_size, num_labels)
    print(f"Classifier after: {model.classifier}")
    model.config.id2label = id2label
    model.config.label2id = label2id
    # Reset problem type to ensure correct loss computation
    model.config.problem_type = None
    X_Train_Val, X_Test, y_Train_Val, y_Test = train_test_split(resume_df_X, resume_df_y, test_size=0.2, stratify=resume_df_y, random_state=42)
    X_Train, X_Val, y_Train, y_Val = train_test_split(X_Train_Val, y_Train_Val, test_size=0.2, stratify=y_Train_Val, random_state=42)

    print(f'size of original training df: {len(X_Train)}')
    Xy_train = pd.DataFrame({"text": X_Train, "label": y_Train})
    Xy_train_augmented = apply_multiple_augmentations(
        Xy_train,
        "text",
        "label", 
        [shuffle_summary, synonym_replace, back_translate],
        ratios=[0.0, 0.0, 0.0])
    X_Train = Xy_train_augmented["text"]
    y_Train = Xy_train_augmented["label"]
    print(f'size of augmented training df: {len(X_Train)}')

    train_dataset, eval_dataset = prepare_data(X_Train, y_Train, X_Val, y_Val)
    finetuned_model = finetune_bert_model(model, train_dataset, eval_dataset)
    # Prepare the test dataset
    test_dataset = Dataset.from_dict({"text": X_Test, "label": y_Test})
    test_dataset = test_dataset.map(preprocess_function, batched=True)



    
