from .graph_utils import count_co_occurences

def textgcn_paper(docs):

    return 

def transductive_graph(docs, method="text gcn paper", **kwargs):
    """
        Turns each doc into a graph,
        returns graphs represented as edge indexes and words
    """
    if method == "text gcn paper":
        graphs = [textgcn_paper(doc, **kwargs) for doc in docs]
        return graphs
    
    raise NotImplementedError("[dataprep] %s inductive graphingis not yet implemented")