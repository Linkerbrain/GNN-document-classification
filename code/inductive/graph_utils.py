from collections import defaultdict

def count_co_occurences(doc, window_size):
    co_occurences=defaultdict(int)

    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[(doc[i], doc[j])] += 1 # Could add weighting based on distance
            else:
                co_occurences[(doc[j], doc[i])] += 1
    return co_occurences
