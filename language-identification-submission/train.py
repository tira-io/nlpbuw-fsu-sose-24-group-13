from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
from preprocessing import clean_text

if __name__ == "__main__":

    # Load the data
    tira = Client()

    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    
    text_train["text"] = clean_text(text_train["text"])
    

    df = text_train.join(targets_train.set_index("id"), on="id")

    model = Pipeline([
       ("vectorizer", CountVectorizer(analyzer='word', ngram_range=(1, 2), max_df=0.99, min_df=0.01)),
       ("classifier", MultinomialNB())
    ])
   
    model.fit(df["text"], df["lang"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")
