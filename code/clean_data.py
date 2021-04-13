import re
import os
import unidecode

# Load

def load_file(filename):
    """
    Loads a tab separated file with a single label
    """
    labels = []
    docs = []

    with open(filename, encoding="utf8", errors="ignore") as f:
        for line in f:
            content = line.split("\t")
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels

# Clean Documents

def make_ascii(s):
    """
    Replaces chars like "ą/ę/ś/ć" with "a/e/s/c".
    This might be bad for some languages but makes it simpler for now
    """
    unaccented_string = unidecode.unidecode(s)

    return unaccented_string

def clean_str(s):
    # Replace all characters not common in English language
    # This might have to be updated later for multilingual support
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    # Pull apart contractions for better matching of words with pretrained embeddings
    s = re.sub(r"\'s", " 's", s)
    s = re.sub(r"\'ve", " 've", s)
    s = re.sub(r"n\'t", " n't", s)
    s = re.sub(r"\'re", " 're", s)
    s = re.sub(r"\'d", " 'd", s)
    s = re.sub(r"\'ll", " 'll", s)
    # Add extra space around punctuation marks
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    # Normalize multiple spaces to a single on
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower().split()

def clean_doc(doc):
    return clean_str(make_ascii(doc))

def clean_docs(docs):
    for i in range(len(docs)):
        docs[i] = clean_doc(docs[i])
    return docs

# Fix Labels

def to_numeric(raw_labels, classes=None):
    if not classes:
        classes = sorted(list(set(raw_labels)))

    mapping = dict(zip(classes, range(0, len(classes))))

    numeric = [mapping[l] for l in raw_labels]

    return numeric


# All together

def get_clean_data(path):
    """
    Combines all dataprep functions,
        loads data
        cleans the strings
        makes labels numeric
    returns
        docs : list of lists of strings
        labels : list of integers
    """

    docs, labels = load_file(path)

    c_docs = clean_docs(docs)
    c_labels = to_numeric(labels)

    return c_docs, c_labels
