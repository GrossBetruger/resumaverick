import pytest
import torch
import pandas as pd

import resumaverick.augmentation as aug

def test_get_synonyms_unknown_pos():
    syns = aug.get_synonyms("anything", "X")
    assert isinstance(syns, list)
    assert syns == []

def test_synonym_replace_no_replace():
    text = "The quick brown fox."
    out = aug.synonym_replace(text, p=0.0)
    assert out == "The quick brown fox . " # only dot and space added

def test_synonym_replace_with_replace():
    text = "The quick brown fox."
    out = aug.synonym_replace(text, p=1.0)
    original_words = text.split()
    augmented_words = out.split()
    assert set(augmented_words) & set(original_words) == {"The"}

def test_apply_multiple_augmentations_basic():
    X = pd.Series(["a", "B"])
    y = pd.Series([1, 2])
    df = pd.DataFrame({"X": X, "y": y})
    def upper(x): return x.upper()
    def lower(x): return x.lower()

    new_df = aug.apply_multiple_augmentations(df=df, x_col="X", y_col="y", augmentations=[upper, lower], verbose=False)
    assert len(new_df) == 3 * len(X)
    assert list(new_df[:2]["X"]) == ["a", "B"]
    assert list(new_df[2:4]["X"]) == ["A", "B"]
    assert list(new_df[4:6]["X"]) == ["a", "b"]
    assert list(new_df["y"]) == [1, 2, 1, 2, 1, 2]

class DummyTokenizer:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.model_max_length = 10
        self.last_text = None

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        self.last_text = texts[0]
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, token_ids, skip_special_tokens):
        return f"{self.src}->{self.tgt}:{self.last_text}"

class DummyModel:
    def generate(self, **batch):
        return batch["input_ids"]

def test_translate_with_dummy():
    tok = DummyTokenizer("en", "fr")
    mod = DummyModel()
    out = aug.translate("hello", tok, mod)
    assert out == "en->fr:hello"

def test_back_translate_with_dummy(monkeypatch):
    def fake_load(src, tgt):
        return DummyTokenizer(src, tgt), DummyModel()
    monkeypatch.setattr(aug, "load_model_and_tokenizer", fake_load)
    result = aug.back_translate("world", intermediate_lang="fr")
    assert result == "fr->en:en->fr:world"
