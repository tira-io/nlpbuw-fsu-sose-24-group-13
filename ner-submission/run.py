from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import json
import evaluate

def load_data():
    tira = Client()
    # Load validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    return text_validation, targets_validation

def prepare_data(text_validation, targets_validation):
    sentences = text_validation['sentence'].apply(lambda x: x.split()).tolist()
    # Generate labels data from the targets_validation
    labels = targets_validation['tags'].tolist()
    
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
    # Load data
    text_validation, targets_validation = load_data()
    
    # Prepare training data
    train_sents = prepare_data(text_validation, targets_validation)
    
    # Extract features and labels
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    
    # Define and train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    # Validate model on the same data (replace with test data for real evaluation)
    X_val = [sent2features(s) for s in train_sents]
    y_pred = crf.predict(X_val)
    
    # Save predictions in the required format
    predictions = pd.DataFrame({'id': text_validation['id'], 'tags': y_pred})
    
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
    
    # Evaluate model using seqeval
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=y_pred, references=y_train)
    print(results)

    # Sklearn metrics for comparison
    labels_list = list(crf.classes_)
    if 'O' in labels_list:
        labels_list.remove('O')
    f1_score = metrics.flat_f1_score(y_train, y_pred, average='weighted', labels=labels_list)
    print(f'Validation F1 Score (sklearn): {f1_score}')
    print(metrics.flat_classification_report(y_train, y_pred, labels=labels_list, digits=3))
