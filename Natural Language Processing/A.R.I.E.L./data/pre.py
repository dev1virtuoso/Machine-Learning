# Copyright (c) 2024 Carson. All rights reserved.

import json

# Read the SQuAD dataset
squad_data_path = "/data/train-v2.0.json"

with open(squad_data_path, "r", encoding="utf-8") as file:
    squad_data = json.load(file)

# Convert the SQuAD dataset to dataset format
dataset = {"data": []}

for article in squad_data["data"]:
    paragraphs = []
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        qas = []
        for qa in paragraph["qas"]:
            question = qa["question"]
            qas.append({"question": question, "answers": []})
        paragraphs.append({"context": context, "qas": qas})
    dataset["data"].append({"paragraphs": paragraphs})

# Save the dataset as a JSON file
dataset_path = "/data/squad_dataset.json"

with open(dataset_path, "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=4)

print("SQuAD dataset has been converted to dataset format and saved as squad_dataset.json")
