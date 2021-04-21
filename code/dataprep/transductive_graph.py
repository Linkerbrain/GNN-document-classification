from .graph_utils import count_co_occurences
from .vocab import create_idx_mapping
from collections import Counter, defaultdict
from math import log

def docname(i):
    return "_DOC"+str(i)+"_"

def textgcn_paper(docs, window_size=4):
    all_words = [word for doc in docs for word in doc]
    all_unique_words = list(set(all_words))
    doc_names = [docname(i) for i in range(len(docs))]

    nodes = all_unique_words+doc_names
    word2idx = create_idx_mapping(nodes)

    rows = []
    columns = []
    weights = []

    ### Word to word edges based on PMI

    # count how many times words occur
    freq = Counter(all_words)
    word_count = len(all_words)
    # count how many times words occur together
    total_window_count = 0
    co_occurences = defaultdict(int)
    for doc in docs:
        co_occurences, window_count = count_co_occurences(doc, window_size=window_size, return_count=True, co_occurences=co_occurences)
        total_window_count += window_count

    # combine and calculate pmi
    for ((word_a, word_b), count) in co_occurences.items():
        # pmi = log ( p(x,y) / (p(x)p(y)) )
        pmi = log((count / total_window_count) / ((freq[word_a] * freq[word_b]) / (word_count ** 2))) # TODO: not tested if correct

        if pmi <= 0:
            continue
        
        word_a_id = word2idx[word_a]
        word_b_id = word2idx[word_b]

        # add twice for symmetry
        rows += [word_a_id, word_b_id]
        columns += [word_b_id, word_a_id]
        weights += [pmi, pmi]

    ### Document to words edges based on TDIDF

    # count in how many docs each word is
    in_document_count = defaultdict(int)
    # count how often each word appears in each doc
    frequency_in_doc = defaultdict(int)

    for i, doc in enumerate(docs):
        already_counted_words = set()
        for word in doc:
            frequency_in_doc[(i, word)] += 1

            if word not in already_counted_words:
                in_document_count[word] += 1
                already_counted_words.add(word)

    doc_count = len(docs)
    # save the edges
    for ((doc_idx, word), count) in frequency_in_doc.items():
        # tf = count in doc / total doc length
        tf = count / len(docs[doc_idx])
        # idf = log (N / docs with that word)
        idf = log(doc_count / in_document_count[word])

        tfidf = tf * idf

        # add twice for symmetry
        doc_id = word2idx[docname(doc_idx)]
        word_id = word2idx[word]

        rows += [doc_id, word_id]
        columns += [word_id, doc_id]
        weights += [tfidf, tfidf]

    edges = ([rows, columns], weights)

    ### Self edges (maybe not neccesary in Pytorch Geometric (??))

    for i in range(len(nodes)):
        rows.append(i)
        columns.append(i)
        weights.append(1)

    return edges, nodes

def transductive_graph(docs, method="text gcn paper", **kwargs):
    """
        Turns each doc into a graph,
        returns graphs represented as edge indexes and words
    """
    if method == "text gcn paper":
        graphs = [textgcn_paper(doc, **kwargs) for doc in docs]
        return graphs
    
    raise NotImplementedError("[dataprep] %s inductive graphingis not yet implemented")