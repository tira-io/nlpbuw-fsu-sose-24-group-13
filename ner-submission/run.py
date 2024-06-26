from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Simple heuristic function to determine entity type based on common patterns
def simple_heuristic_token_classification(token):
    if token.istitle():
        return "B-per"  # Assume title case words are persons
    elif token.isupper():
        return "B-org"  # Assume upper case words are organizations
    else:
        return "O"  # Default to outside any named entity

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    # labeling the data with simple heuristics
    predictions = text_validation.copy()
    predictions['tags'] = predictions['sentence'].apply(lambda x: [simple_heuristic_token_classification(token) for token in x.split(' ')])
    predictions = predictions[['id', 'tags']]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
