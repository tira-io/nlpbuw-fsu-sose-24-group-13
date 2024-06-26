from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    tira = Client()

    # Load validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Generate predictions (classifying each token as "B-geo")
    predictions = []
    for index, row in text_validation.iterrows():
        sentence = row['sentence']
        tokens = sentence.split()
        tags = ['B-geo'] * len(tokens)
        predictions.append({'id': row['id'], 'tags': tags})

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_file = Path(output_directory) / "predictions.jsonl"
    
    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

    print(f"Predictions saved to {output_file}")
