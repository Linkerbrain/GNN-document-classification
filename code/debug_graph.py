import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

def plot_adjacency(adjacency, word_idx, marker_size=1):
    fig, ax =  plt.subplots(figsize=(12, 12))

    ax.spy(adjacency, markersize=marker_size, color="orange")

    print(word_idx)
    sorted_labels = sorted(word_idx, key=word_idx.get)

    ax.set_xticks(list(word_idx.values()))
    ax.set_xticklabels(word_idx.keys(), rotation = 90)
    ax.set_yticks(list(word_idx.values()))
    ax.set_yticklabels(word_idx.keys(), rotation = 0)

    plt.show()

def vis_graph(adjacency, word_idx):
    G = nx.from_scipy_sparse_matrix(adjacency)

    mapping = {}
    for word, idx in word_idx.items():
        mapping[idx] = word

    G = nx.relabel_nodes(G, mapping)

    net = Network(height = '800px', width = '1200px')

    # for each node and its attributes in the networkx graph
    for node,node_attrs in G.nodes(data=True):
        net.add_node(str(node),**node_attrs)
        
    # for each edge and its attributes in the networkx graph
    for source,target,edge_attrs in G.edges(data=True):
        # if value/width not specified directly, and weight is specified, set 'value' to 'weight'
        if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
            # place at key 'value' the weight of the edge
            edge_attrs['value']=edge_attrs['weight']
        # add the edge
        net.add_edge(str(source),str(target),**edge_attrs)

    net.show_buttons()
    net.show("example.html")