from collections import defaultdict
import enum
import scipy.sparse as sp

# Inductive classification

def count_co_occurences(doc, window_size):
    co_occurences = defaultdict(int)
    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[(doc[i], doc[j])] += 1 # / Could add weighting
            else:
                co_occurences[(doc[j], doc[i])] += 1
    return co_occurences

def graph_unique_coocc(doc, window_size=3):
    """
    Creates graph:
        vertex for each unique word
        edge with co-occurence count
    Returns graph as:
        adjacency matrix
        dictionary with what words belong to which index of matrix
    """
    # make nodes
    unique_words = list(set(doc))
    word2idx = {word: ix for ix, word in enumerate(unique_words)}
    idx2word = {ix: word for ix, word in enumerate(unique_words)}

    # make edges
    co_occurences = count_co_occurences(doc, window_size)

    rows = []
    columns = []
    weights = []
    for ((word_a, word_b), count) in co_occurences.items():
        word_a_id = word2idx[word_a]
        word_b_id = word2idx[word_b]

        # add twice for symmetry
        rows += [word_a_id, word_b_id]
        columns += [word_b_id, word_a_id]
        weights += [count, count]

    adjacency = sp.csr_matrix((weights, (rows, columns)), shape=(len(unique_words), len(unique_words)),
                                dtype=float)

    # TODO: Normalize?

    return adjacency, unique_words

def add_master_node(adjacency, idx2word):
    """
    Adds master node to graph

    connected to everything

    TODO the master node needs to be added to vocab for pipeline to work
    TODO master node as first node since zeroes are padded at back
    """
    # make node
    idx2word[len(idx2word)] = "MASTER NODE"

    # add edges
    adjacency = sp.hstack([adjacency, sp.csr_matrix([[1] for _ in adjacency])])
    adjacency = sp.vstack([adjacency, sp.csr_matrix([[1 for _ in range(adjacency.shape[1])]])])

    return adjacency, idx2word

# Transductive classification

# TODO later

# notes
# dataloader dubbel copy
# window size 1 kleiner dan verwacht
# collate no master returned verkeerde nodes
# 