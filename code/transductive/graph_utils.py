from collections import defaultdict

def count_co_occurences(doc, window_size, return_count=False, co_occurences=defaultdict(int)):
    window_count = 0
    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            window_count += 1
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[(doc[i], doc[j])] += 1 # Could add weighting based on distance
            else:
                co_occurences[(doc[j], doc[i])] += 1

    if return_count:
        return co_occurences, window_count
    return co_occurences
