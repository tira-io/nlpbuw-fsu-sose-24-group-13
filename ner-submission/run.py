import json
import pandas as pd
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from seqeval.metrics import classification_report

def load_data():
    tira = Client()
    text_data = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_data = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    data = []
    for i in range(len(text_data)):
        tokens = text_data.iloc[i]['sentence'].split()
        tags = targets_data.iloc[i]['tags']
        data.append((tokens, tags))

    return data

def simple_rule_based_ner(sentence):
    tags = []
    for word in sentence:
        if word.istitle():
            tags.append('B-per')
        else:
            tags.append('O')
    return tags

def save_predictions(predictions, output_path):
    output_data = []
    for idx, pred in enumerate(predictions):
        output_data.append({"id": idx, "tags": pred})
    with open(output_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    # Load and prepare data
    data = load_data()

    # Predict using rule-based NER
    predictions = []
    true_labels = []
    for tokens, tags in data:
        pred_tags = simple_rule_based_ner(tokens)
        predictions.append(pred_tags)
        true_labels.append(tags)

    # Evaluate
    print(classification_report(true_labels, predictions))

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    save_predictions(predictions, Path(output_directory) / "predictions.jsonl")
