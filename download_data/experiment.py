from datasets import load_dataset

LANGUAGE = "en"

AMOUNT = 10000

OUTPUT = "C:/Users/Lodewijk/Desktop/scriptie/GNN-document-classification/data/amazon_pos_neg.train.10000.en"

# gather balanced subset
amount_per_star = AMOUNT // 2
dataset = load_dataset("amazon_reviews_multi", LANGUAGE)
dataset["train"].shuffle()

chosen_entries = {"POS ":[], "NEG ":[]}
for entry in dataset["train"]:
    stars = entry["stars"]

    if stars < 3:
        cat = "POS "

        if len(chosen_entries[cat]) < amount_per_star:
            chosen_entries[cat].append(entry["review_body"])
    if stars > 3:
        cat = "NEG "

        if len(chosen_entries[cat]) < amount_per_star:
            chosen_entries[cat].append(entry["review_body"])

# save as tsv
with open(OUTPUT, 'w', encoding="utf8") as output_f:
    for stars, reviews in chosen_entries.items():
        for review in reviews:
            output_f.write(stars + "\t" + review + "\n")
