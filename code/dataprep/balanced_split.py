from random import shuffle

def get_balanced_split(docs, labels, train_amount, test_amount):
    """
    Ugly implementation but first thing that came to mind
    """
    unique_labels = list(set(labels))
    train_amount_per_label = train_amount // len(unique_labels)
    test_amount_per_label = test_amount // len(unique_labels)
    
    data = list(zip(docs, labels))

    shuffle(data)

    train_docs = []
    train_labels = []

    label_counts = {label : 0 for label in unique_labels}
    done = 0
    onwards = 0
    for i, (doc, label) in enumerate(data):
        if label_counts[label] < train_amount_per_label:
            label_counts[label] += 1
            train_docs.append(doc)
            train_labels.append(label)
            continue
        
        if label_counts[label] == train_amount_per_label:
            label_counts[label] += 1
            done += 1
            if done == len(unique_labels):
                onwards = i
                break

    test_docs = []
    test_labels = []
    label_counts = {label : 0 for label in unique_labels}
    done = 0
    for (doc, label) in data[onwards+1:]:
        if label_counts[label] < test_amount_per_label:
            label_counts[label] += 1
            test_docs.append(doc)
            test_labels.append(label)
            continue
        
        if label_counts[label] == test_amount_per_label:
            label_counts[label] += 1
            if done == len(unique_labels):
                break

    return train_docs, train_labels, test_docs, test_labels