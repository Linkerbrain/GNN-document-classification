import numpy as np
from collections import Counter, OrderedDict
from gensim.models.keyedvectors import KeyedVectors
# process corpus

def make_vocab(docs, min_count = 2):
    custom_tokens = ["__OOV__"]
    word_counter = Counter([word for doc in docs for word in doc] + custom_tokens)

    vocab = []
    for i, (word, count) in enumerate(word_counter.items()):
        if count >= min_count:
            vocab.append(word)


    print("[dataprep] Found vocab of %d words (occuring at least %d times)" % (len(vocab), min_count))

    return vocab + custom_tokens

#   EMBEDDINGS

def word2vec_embed(vocab, w2v_file):
    model = KeyedVectors.load_word2vec_format(w2v_file, binary=True)

    embedding = {}
    fail_rate = 0
    for word in vocab:
        if word in model:
            embedding[word] = model[word]
        else:
            fail_rate += 1
            embedding[word] = np.random.uniform(-0.25, 0.25, 300)

    print("[dataprep] Word2Vec embedding succesfully encoded %d/%d words in vocabulary"
            % (len(vocab)-fail_rate, len(vocab)))

    return embedding

def index_embed(vocab):
    embedding = {}

    vector_length = len(vocab)

    for i, word in enumerate(vocab):
        embedding[word] = np.array([i])
    print("[dataprep] Index-embedded with dimension %d" % (vector_length))

    return embedding

def onehot_embed(vocab):
    embedding = {}

    vector_length = len(vocab)

    for i, word in enumerate(vocab):
        vec = np.zeros(vector_length)
        vec[i] = 1
        embedding[word] = vec

    print("[dataprep] Onehot-embedded with dimension %d" % (vector_length))

    return embedding

def embed_vocab(vocab, embedding_type="index", **kwargs):
    if embedding_type == "word2vec":
        return word2vec_embed(vocab, **kwargs)
    if embedding_type == "onehot":
        return onehot_embed(vocab)

    return index_embed(vocab)

# process labels

def make_label_mapping(labels):
    classes = sorted(list(set(labels)))

    mapping = {label : int(i) for i, label in enumerate(classes)}
    print(mapping)

    return mapping

# All together

def make_mappings(labels, docs, embedding_type="index", **kwargs):
    vocab = make_vocab(docs)

    word2embedding = embed_vocab(vocab, embedding_type, **kwargs)

    label2number = make_label_mapping(labels)

    return label2number, word2embedding
