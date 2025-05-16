import spacy
from nltk.corpus import wordnet
import random

import nltk

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


if __name__ == "__main__":
    text = "Led a team of skilled aviation engineers to develop innovative flight solutions."
    aug_text = synonym_replace(text, p=0.3)
    print("Original: ", text)
    print("Augmented:", aug_text)
