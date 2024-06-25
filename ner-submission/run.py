import matplotlib.pyplot as plt
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
import json
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split

# Set the style for plotting
plt.style.use('ggplot')

def load_data():
    tira = Client()
    # Loading validation data (automatically replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    return text_validation, targets_validation

def prepare_data(text_validation, targets_validation):
    sentences = text_validation['sentence'].apply(lambda x: x.split()).tolist()
    labels = targets_validation['tags'].tolist()
    
    
    # Generate mock 'tags' data similar to the baseline approach
    labels = [['B-geo'] * len(sentence) for sentence in sentences]

    # Combine words and labels into a single structure

    # Generate mock 'tags' data similar to the baseline approach
    labels = [['B-geo'] * len(sentence) for sentence in sentences]

    # Combine words and labels into a single structure
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
    # Add features for the previous word
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    # Add features for the next word
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
    # Initialize TIRA client
    tira = Client()
    # Load validation data
    text_validation, targets_validation = load_data()

    # Prepare data for training
    train_sents = prepare_data(text_validation, targets_validation)

    # Split the data into training and validation sets
    train_sents, val_sents = train_test_split(train_sents, test_size=0.2, random_state=42)

    # Convert sentences to features and labels for training
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    # Initialize and train the CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Convert sentences to features and labels for validation
    X_val = [sent2features(s) for s in val_sents]
    y_val = [sent2labels(s) for s in val_sents]

    # Predict on the validation data
    y_pred = crf.predict(X_val)

    # Flatten predictions and actual labels for alignment with text_validation
    flat_y_pred = [tag for sent in y_pred for tag in sent]
    flat_text_ids = text_validation['id'].repeat(text_validation['sentence'].apply(lambda x: len(x.split()))).values

    # Create DataFrame for saving predictions
    predictions = pd.DataFrame({'id': flat_text_ids[:len(flat_y_pred)], 'tags': flat_y_pred})

    # Save predictions to the output directory
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True, force_ascii=False)

    # Evaluate using seqeval
    print(classification_report(y_val, y_pred, zero_division=0))

    # Calculate sklearn metrics for comparison
    labels_list = list(crf.classes_)
    if 'O' in labels_list:
        labels_list.remove('O')
    f1_score = metrics.flat_f1_score(y_val, y_pred, average='weighted', labels=labels_list, zero_division=0)
    print(f'Validation F1 Score (sklearn): {f1_score}')
    print(metrics.flat_classification_report(y_val, y_pred, labels=labels_list, digits=3, zero_division=0))
