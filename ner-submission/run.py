import json
import pandas as pd
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

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
        data.append([(token, tag) for token, tag in zip(tokens, tags)])

    return data

def train_crf_model(train_data):
    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )
    crf.fit(X_train, y_train)
    return crf

def predict(crf, sentences):
    X_test = [sent2features(s) for s in sentences]
    y_pred = crf.predict(X_test)
    return y_pred

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
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train CRF model
    crf = train_crf_model(train_data)

    # Predict on validation data
    val_sentences = [sent2tokens(s) for s in val_data]
    y_pred = predict(crf, val_sentences)

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    save_predictions(y_pred, Path(output_directory) / "predictions.jsonl")
