from pathlib import Path
from joblib import dump
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from tira.rest_api_client import Client

def preprocess_data(text_data, labels_data):
    data = []
    for i in range(len(text_data)):
        sentence = text_data.iloc[i]['sentence'].split()
        labels = labels_data.iloc[i]['tags']
        data.append((sentence, labels))
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

def sent2labels(sentence):
    return [label for label in sentence]

if __name__ == "__main__":
    tira = Client()

    # Load the data
    text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-training-20240612-training")
    targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-training-20240612-training")

    # Preprocess data
    train_data = preprocess_data(text_train, targets_train)
    X_train = [sent2features(s) for s, t in train_data]
    y_train = [t for s, t in train_data]

    # Train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Save the model
    dump(crf, Path(__file__).parent / "model.joblib")
