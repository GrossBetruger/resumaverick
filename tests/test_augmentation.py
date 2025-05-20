import pytest
import torch
import pandas as pd
from itertools import combinations
import resumaverick.augmentation as aug


def test_get_synonyms_unknown_pos():
    syns = aug.get_synonyms("anything", "X")
    assert isinstance(syns, list)
    assert syns == []


def test_synonym_replace_no_replace():
    text = "The quick brown fox."
    out = aug.synonym_replace(text, p=0.0)
    assert out == "The quick brown fox . "  # only dot and space added


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

    def upper(x):
        return x.upper()

    def lower(x):
        return x.lower()

    new_df = aug.apply_multiple_augmentations(
        df=df, x_col="X", y_col="y", augmentations=[upper, lower], verbose=False
    )
    assert len(new_df) == 3 * len(X)
    assert list(new_df[:2]["X"]) == ["a", "B"]
    assert list(new_df[2:4]["X"]) == ["A", "B"]
    assert list(new_df[4:6]["X"]) == ["a", "b"]
    assert list(new_df["y"]) == [1, 2, 1, 2, 1, 2]


def test_apply_multiple_augmentations_with_ratios():
    X = pd.Series(["a", "B", "c", "D"])
    y = pd.Series(["ya", "yb", "yc", "yd"])
    df = pd.DataFrame({"X": X, "y": y})

    def upper(x):
        return x.upper()

    def lower(x):
        return x.lower()

    def add_dot(x):
        return x + "."

    ratios = [0.5, 0.25, 1]
    new_df = aug.apply_multiple_augmentations(
        df=df,
        x_col="X",
        y_col="y",
        augmentations=[upper, lower, add_dot],
        ratios=ratios,
        verbose=False,
    )
    assert len(new_df) == 11  # 4 + 4*0.5 + 4*0.25 + 4*1 = 11
    assert list(new_df[:4]["X"]) == ["a", "B", "c", "D"]
    # ratio less than 1 are unordered
    upper_X = X.apply(upper).tolist()
    # 0.5 ratio, 2 augmented items
    assert tuple(new_df[4:6]["X"]) in list(combinations(upper_X, 2))
    # 0.25 ratio, 1 augmented items
    lower_X = X.apply(lower).tolist()
    assert tuple(new_df[6:7]["X"]) in list(combinations(lower_X, 1))
    # 1 ratio, 8 augmented items
    dot_X = X.apply(add_dot).tolist()
    assert set(new_df[7:15]["X"]) == set(dot_X)

    # test target match:
    # new_df["x_all_lower"] = new_df["X"].apply(lower)
    grouped = new_df.groupby("X")["y"].unique()
    # 1️⃣  Each lowercase X has a single unique y
    assert (grouped.apply(len) == 1).all()
    # 2️⃣  …and that y matches the original mapping
    for x, y in grouped.apply(lambda arr: arr[0]).to_dict().items():
        assert y == "y" + x.lower()[0], f"y={y} does not match X: {x}"


def test_apply_multiple_augmentations_with_0_ratios():
    X = pd.Series(["a", "B"])
    y = pd.Series(["ya", "yb"])
    df = pd.DataFrame({"X": X, "y": y})

    def upper(x):
        return x.upper()

    def lower(x):
        return x.lower()

    new_df = aug.apply_multiple_augmentations(
        df=df,
        x_col="X",
        y_col="y",
        augmentations=[upper, lower],
        ratios=[1, 0.0],
        verbose=False,
    )
    assert len(new_df) == 4  # 2 + 2*1 + 2*0.0
    assert list(new_df["X"]) == ["a", "B", "A", "B"]

    no_aug_df = aug.apply_multiple_augmentations(
        df=df,
        x_col="X",
        y_col="y",
        augmentations=[upper, lower],
        ratios=[0.0, 0.0],
        verbose=False,
    )
    assert list(no_aug_df["X"]) == ["a", "B"]


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


def test_sentence_shuffle():
    text = "I went home. This is it, What are the odds? premutations are fun. Almost surely to pass."
    out = aug.sentence_shuffle(text)
    assert out != text
    for sent in text.split(" "):
        assert sent in out, f"'{sent}' not in '{out}'"


def test_sentence_shuffle_with_summary():
    text = "Blah blah blah. Summary: Seasoned software engineer. Likes to code. Has a PhD in math, and many hobbies. Experience: 10 years of experience in software engineering."
    out = aug.shuffle_summary(text, seed=42)
    assert (
        out
        == "Blah blah blah. Summary: Likes to code.  Seasoned software engineer. Has a PhD in math, and many hobbies.Experience: 10 years of experience in software engineering."
    )
