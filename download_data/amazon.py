from datasets import load_dataset

LANGUAGE = "en"

AMOUNT = 10000

OUTPUT = "C:/Users/Lodewijk/Desktop/scriptie/GNN-document-classification/data/amazon.train.10000.en"

# gather balanced subset
amount_per_star = AMOUNT // 5
dataset = load_dataset("amazon_reviews_multi", LANGUAGE)
dataset["train"].shuffle()

chosen_entries = {1:[], 2:[], 3:[], 4:[], 5:[]}
for entry in dataset["train"]:
    if len(chosen_entries[entry["stars"]]) < amount_per_star:
        chosen_entries[entry["stars"]].append(entry["review_body"])

# save as tsv
with open(OUTPUT, 'w', encoding="utf8") as output_f:
    for stars, reviews in chosen_entries.items():
        for review in reviews:
            output_f.write(str(stars) + "\t" + review + "\n")
