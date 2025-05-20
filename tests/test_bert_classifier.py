import pytest
import numpy as np
import pandas as pd

from resumaverick import bert_classifier as bc


def test_compute_metrics_perfect_prediction():
    # Two samples, three classes
    logits = np.array([[0.1, 0.8, 0.1],
                       [0.9, 0.05, 0.05]])
    labels = np.array([1, 0])
    metrics = bc.compute_metrics((logits, labels))
    assert pytest.approx(metrics["accuracy"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["f1"], rel=1e-6) == 1.0


def test_load_bert_model_monkeypatched(monkeypatch):
    # Monkeypatch AutoTokenizer and AutoModelForSequenceClassification
    class DummyTok:
        @staticmethod
        def from_pretrained(name):
            return f"tok-{name}"
        
    class DummyMod:
        @staticmethod
        def from_pretrained(name):
            return f"mod-{name}"
    monkeypatch.setattr(bc, "AutoTokenizer", DummyTok)
    monkeypatch.setattr(bc, "AutoModelForSequenceClassification", DummyMod)
    tok, mod = bc.load_bert_model("my-model")
    assert tok == "tok-my-model"
    assert mod == "mod-my-model"


def test_preprocess_and_prepare_data(monkeypatch):
    # Dummy tokenizer for preprocessing
    class DummyTok:
        def __call__(self, texts, truncation, padding, max_length):
            batch_size = len(texts)
            return {
                "input_ids": [[42] * max_length for _ in range(batch_size)],
                "attention_mask": [[1] * max_length for _ in range(batch_size)],
            }
    monkeypatch.setattr(bc, "tokenizer", DummyTok())
    # Test preprocess_function
    examples = {"text": ["a", "b"]}
    out = bc.preprocess_function(examples)
    assert "input_ids" in out and "attention_mask" in out
    assert isinstance(out["input_ids"], list)
    assert len(out["input_ids"]) == 2
    # Test prepare_data
    train_X = pd.Series(["one", "two"])
    train_y = pd.Series([0, 1])
    val_X = pd.Series(["three"])
    val_y = pd.Series([0])
    train_ds, eval_ds = bc.prepare_data(train_X, train_y, val_X, val_y)
    for ds, expected_rows in [(train_ds, 2), (eval_ds, 1)]:
        assert "input_ids" in ds.column_names
        assert "attention_mask" in ds.column_names
        assert "label" in ds.column_names
        assert ds.num_rows == expected_rows