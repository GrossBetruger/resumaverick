import spacy
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import random

import nltk
from tqdm import tqdm

tqdm.pandas()
# Download NLTK resources only when running as a script
if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")

def get_synonyms(word, pos_tag):
    """Return a list of synonyms for a word, filtered by POS tag."""
    tag_map = {
        "NOUN": wordnet.NOUN,
        "VERB": wordnet.VERB,
        "ADJ": wordnet.ADJ,
        "ADV": wordnet.ADV
    }
    wn_tag = tag_map.get(pos_tag, None)
    if wn_tag is None:
        return []
    synonyms = set()
    for syn in wordnet.synsets(word, pos=wn_tag):
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace("_", " ")
            if lemma_name.lower() != word.lower():
                synonyms.add(lemma_name)
    return list(synonyms)

def synonym_replace(text, p=0.2):
    """Randomly replace words in the text with synonyms."""
    doc = nlp(text)
    new_tokens = []
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"} and random.random() < p:
            syns = get_synonyms(token.text, token.pos_)
            if syns:
                new_tokens.append(random.choice(syns))
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
    return spacy.tokens.doc.Doc(doc.vocab, words=new_tokens).text


def load_model_and_tokenizer(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(text, tokenizer, model):
    # Tokenize with truncation to avoid exceeding model's max position embeddings
    batch = tokenizer([
        text
    ], return_tensors="pt", padding=True, truncation=True,
        max_length=tokenizer.model_max_length)
    gen = model.generate(**batch)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

def back_translate(text, intermediate_lang='fr'):
    # English to intermediate
    tok_en_fr, mod_en_fr = load_model_and_tokenizer('en', intermediate_lang)
    fr_text = translate(text, tok_en_fr, mod_en_fr)
    # Intermediate back to English
    tok_fr_en, mod_fr_en = load_model_and_tokenizer(intermediate_lang, 'en')
    back_text = translate(fr_text, tok_fr_en, mod_fr_en)
    return back_text


def apply_multiple_augmentations(df: pd.DataFrame, x_col: str, y_col: str, augmentations: list[callable], verbose: bool = True, ratios: list[float] = None) -> tuple[pd.Series, pd.Series]:

    """
    Apply multiple augmentations to the text.
    """
    augmented_dfs = []
    for i, augmentation in enumerate(augmentations):
        ratio = ratios[i] if ratios else 1
        if ratio > 1:
            sample = df.sample(frac=ratio, random_state=42)
        else:
            sample = df
        if verbose:
            print(f'applying {augmentation.__name__} to {len(sample)} resumes')
        aug_x = sample[x_col].progress_apply(augmentation)
        aug_y = sample[y_col]
        aug_df = pd.DataFrame({x_col: aug_x, y_col: aug_y})
        augmented_dfs.append(aug_df)
    return pd.concat([df] + augmented_dfs)


if __name__ == "__main__":
    text = "Led a team of skilled aviation engineers to develop innovative flight solutions."
    aug_text = synonym_replace(text, p=0.3)
    print("Original: ", text)
    print("Augmented:", aug_text)

    back_translated = back_translate(text)
    print("Original:        ", text)
    print("Back-translated: ", back_translated)

