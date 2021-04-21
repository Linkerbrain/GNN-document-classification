import numpy as np
from collections import Counter, OrderedDict

def create_idx_mapping(list):
    index_mapping = {}

    for i, element in enumerate(list):
        index_mapping[element] = i

    return index_mapping

def build_word_vocab(docs, min_count=1, extra_tokens=["___UNK___"]):
    all_words = [word for doc in docs for word in docs]

    if min_count == 1:
        unique_words = list(set(all_words)) + extra_tokens
        return create_idx_mapping(unique_words)

    word_counter = Counter([word for doc in docs for word in doc])

    common_words = []
    for i, (word, count) in enumerate(word_counter.items()):
        if count >= min_count:
            common_words.append(word)

    return create_idx_mapping(common_words + extra_tokens)

def build_label_vocab(labels):
    classes = sorted(list(set(labels)))

    return create_idx_mapping(classes)

def vocab(docs, labels, min_word_count=1):
    """
    Creates a vocab of the documents and labels
        a vocab is represented as a mapping from element to index
        (for quick parsing)
    """
    word_vocab = build_word_vocab(docs, min_word_count)
    label_vocab = build_label_vocab(labels)

    return word_vocab, label_vocab
