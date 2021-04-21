from .graph_utils import count_co_occurences

def unique_co_occurence(doc, window_size=3):
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

    # adjacency = sp.csr_matrix((weights, (rows, columns)), shape=(len(unique_words), len(unique_words)),
    #                             dtype=float)

    edges = ([rows, columns], weights)

    return edges, unique_words

def inductive_graph(docs, method="unique co-occurence", **kwargs):
    """
        Turns each doc into a graph,
        returns graphs represented as edge indexes and words
    """
    if method == "unique co-occurence":
        graphs = [unique_co_occurence(doc, **kwargs) for doc in docs]
        return graphs
    
    raise NotImplementedError("[dataprep] %s inductive graphingis not yet implemented")