import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
import re
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words]

def sentence_to_embedding(sentence, word2vec_model):
    # print(sentence)
    words = preprocess_text(sentence)
    embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not embeddings:  # Handle case where no words are in vocabulary
        return np.zeros(word2vec_model.vector_size)
    return np.mean(embeddings, axis=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))