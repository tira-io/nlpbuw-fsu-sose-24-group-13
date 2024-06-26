import json
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def generate_predictions(text_validation):
    predictions = []
    for _, row in text_validation.iterrows():
        sentence = row['sentence']
        tokens = sentence.split()
        tags = ['B-geo'] * len(tokens)  # Simplified tagging logic
        predictions.append({'id': int(row['id']), 'tags': tags})
    return predictions

if __name__ == "__main__":
    tira = Client()

    # Load validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Generate predictions
    predictions = generate_predictions(text_validation)

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_file = Path(output_directory) / "predictions.jsonl"

    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    print(f"Predictions saved to {output_file}")
