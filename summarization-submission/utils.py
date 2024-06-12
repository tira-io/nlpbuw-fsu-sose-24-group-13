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