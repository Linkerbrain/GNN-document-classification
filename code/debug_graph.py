import matplotlib.pyplot as plt

def plot_graph(adjacency, word_idx, marker_size=1):
    fig, ax =  plt.subplots(figsize=(12, 12))

    ax.spy(adjacency, markersize=marker_size, color="orange")

    print(word_idx)
    sorted_labels = sorted(word_idx, key=word_idx.get)

    ax.set_xticks(list(word_idx.values()))
    ax.set_xticklabels(word_idx.keys(), rotation = 90)
    ax.set_yticks(list(word_idx.values()))
    ax.set_yticklabels(word_idx.keys(), rotation = 0)

    plt.show()