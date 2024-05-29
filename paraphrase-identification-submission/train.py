
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer

from tira.rest_api_client import Client
from gensim.models import Word2Vec
import gensim
import gensim.downloader as api

from utils import * 


if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")


    sentences = []
    for sentence1, sentence2 in zip(text['sentence1'], text['sentence2']):
        sentences.append(preprocess_text(sentence1))
        sentences.append(preprocess_text(sentence2))


    word2vec_model = Word2Vec(sentences, vector_size=1000, window=7, min_count=1, workers=4, sg=0, epochs=1000)
    
    # corpus = [' '.join(sent) for sent in sentences]
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))


    # # Update embeddings with frequency weights
    # new_word_vectors = [(word, word2vec_model.wv[word] * idf_dict[word]) for word in word2vec_model.wv.key_to_index.keys() if word in idf_dict]
    # new_word_vectors_dict = {word: vector for word, vector in new_word_vectors}
    # word_freq_dict = {word: word2vec_model.wv.key_to_index[word] for word in new_word_vectors_dict.keys()}

    # # Create a new Word2Vec model with modified vectors
    # word2vec_model_with_weights = Word2Vec(vector_size=1000, window=7, min_count=1, workers=4, sg=0)
    # word2vec_model_with_weights.build_vocab_from_freq(word_freq_dict)
    # word2vec_model_with_weights.wv.vectors = np.array([new_word_vectors_dict[word] for word in word2vec_model_with_weights.wv.index_to_key])
    
    # word2vec_model = word2vec_model_with_weights
    # print(word2vec_model.wv.most_similar('person'))

    sentence1_embeddings = [sentence_to_embedding(item, word2vec_model) for item in text['sentence1']]
    sentence2_embeddings = [sentence_to_embedding(item, word2vec_model) for item in text['sentence2']]
    similarities = [cosine_similarity(a, b) for a, b in zip(sentence1_embeddings, sentence2_embeddings)]
    text['sim'] = similarities

    df = text.join(labels)
    # print(df.head(20))

    model_path = "paraphrase-identification-submission/model.bin"
    word2vec_model.save(model_path)

    thresholds = np.arange(0, 1, 0.001)
    best_threshold = 0
    best_mcc = -1

    for threshold in thresholds:
        predictions = (df['sim'] >= threshold).astype(int)
        mcc = matthews_corrcoef(df['label'], predictions)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    predictions = (df['sim'] >= best_threshold).astype(int)
    accuracy = accuracy_score(df['label'], predictions)
    mcc = matthews_corrcoef(df['label'], predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Best threshold: {best_threshold}")