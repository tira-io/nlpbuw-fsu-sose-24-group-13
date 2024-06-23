import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import chain
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path

def load_data():
    tira = Client()

    # Loading validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    
    print("Text validation columns:", text_validation.columns)
    print("Targets validation columns:", targets_validation.columns)
    print("Sample data from targets_validation:", targets_validation.head())
    
    return text_validation, targets_validation

def prepare_data(text_validation, targets_validation):
    # Check for the correct column name
    if 'sentence' in text_validation.columns:
        sentences = text_validation['sentence'].apply(lambda x: x.split()).tolist()
    elif 'text' in text_validation.columns:
        sentences = text_validation['text'].apply(lambda x: x.split()).tolist()
    else:
        raise KeyError("Neither 'sentence' nor 'text' column found in text_validation DataFrame")
    
    if 'tags' in targets_validation.columns:
        labels = targets_validation['tags'].tolist()
    elif 'text' in targets_validation.columns:
        print("Warning: 'tags' column not found in targets_validation. Using 'text' as labels, which is not expected.")
        labels = targets_validation['text'].apply(lambda x: x.split()).tolist()
    else:
        raise KeyError("The expected 'tags' column was not found in targets_validation DataFrame")
    
    # Combine sentences and labels into a format suitable for feature extraction
    train_sents = []
    for sent, label in zip(sentences, labels):
        train_sents.append(list(zip(sent, label)))
    
    return train_sents

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
    return [label for word, label in sent]

def sent2tokens(sent):
    return [word for word, label in sent]

if __name__ == "__main__":
    text_validation, targets_validation = load_data()
    train_sents = prepare_data(text_validation, targets_validation)

    # Feature extraction
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    # Initialize and train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Predict on validation data
    X_val = [sent2features(s) for s in train_sents]
    y_pred = crf.predict(X_val)

    # Prepare predictions for saving
    predictions = pd.DataFrame({'id': text_validation['id'], 'tags': y_pred})

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True, force_ascii=False)

    # Evaluation (optional)
    labels_list = list(crf.classes_)
    labels_list.remove('O')
    f1_score = metrics.flat_f1_score(y_train, y_pred, average='weighted', labels=labels_list)
    print(f'Validation F1 Score: {f1_score}')
    print(metrics.flat_classification_report(y_train, y_pred, labels=labels_list, digits=3))
