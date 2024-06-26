from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

def preprocess_data(text_data):
    data = []
    for i in range(len(text_data)):
        sentence = text_data.iloc[i]['sentence'].split()
        data.append(sentence)
    return data

def extract_features(sentence, i):
    word = sentence[i]
    features = {
        'word': word,
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
        'suffix-3': word[-3:],
    }
    if i > 0:
        word1 = sentence[i-1]
        features.update({
            '-1:word': word1,
            '-1:is_upper': word1.isupper(),
            '-1:is_title': word1.istitle(),
            '-1:is_digit': word1.isdigit(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence)-1:
        word1 = sentence[i+1]
        features.update({
            '+1:word': word1,
            '+1:is_upper': word1.isupper(),
            '+1:is_title': word1.istitle(),
            '+1:is_digit': word1.isdigit(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]

if __name__ == "__main__":
    tira = Client()

    # Load the data
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Preprocess data
    val_data = preprocess_data(text_validation)
    X_val = [sent2features(s) for s in val_data]

    # Load the model
    model = load(Path(__file__).parent / "model.joblib")

    # Predict
    y_pred = model.predict(X_val)

    # Save predictions
    predictions = text_validation.copy()
    predictions['tags'] = [list(x) for x in y_pred]
    predictions = predictions[['id', 'tags']]
    
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
