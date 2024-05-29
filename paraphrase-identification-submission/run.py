from pathlib import Path
from gensim.models import Word2Vec
import os
import json

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

from utils import *

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")
    # print(df.head(10))  
    model_file = "model.bin"
    model_path = os.path.join(str(Path(__file__).parent), model_file)
    word2vec_model = Word2Vec.load(model_path)


    threshold = 0.827 # imperically defined

    sentence1_embeddings = [sentence_to_embedding(item, word2vec_model) for item in df['sentence1']]
    sentence2_embeddings = [sentence_to_embedding(item, word2vec_model) for item in df['sentence2']]
    df["similarities"] = [cosine_similarity(a, b) for a, b in zip(sentence1_embeddings, sentence2_embeddings)]
    df["label"] = (df["similarities"] >= threshold).astype(int)
    
    df = df.drop(columns=["similarities", "sentence1", "sentence2"]).reset_index()
    # print(df['label'].value_counts())
    # Save the predictions to a JSON file
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
