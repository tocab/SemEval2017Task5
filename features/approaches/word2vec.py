from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVR
import itertools


def word2vec_approach(features, y, train, test):
    """
    Word2Vec approach for sentiment analysis.
    Source: https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
    :param df: DataFrame with text
    :param train: train IDs
    :param test: test IDs
    :param textcolumn: the column that is used for text processing
    :return: score
    """

    # Use Support vector machine regression_classification
    clf = SVR().fit(features[train], y[train])

    # Fit
    clf.fit(features[train], y[train])

    # Predict
    predicted = clf.predict(features[test])

    # Calculate cosine similarity
    cos = cosine_similarity(predicted.reshape(1, -1), y[test].reshape(1, -1))
    return predicted, cos


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, w2v, matrix, avg_len=0):
    word_vectors = []
    for word in text:
        try:
            word_vectors.append(w2v[word])
        except KeyError:
            word_vectors.append(np.zeros(size))
            continue

    if len(text) <= 0:
        word_vectors.append(np.zeros(size))

    if matrix:
        new_token_vector = []
        for cycle_count, cycle in enumerate(itertools.cycle(word_vectors)):
            new_token_vector.append(cycle)
            if cycle_count == avg_len:
                break

        new_token_vector = np.asarray(new_token_vector)
        vec = new_token_vector.flatten()

    else:
        vec = np.mean(word_vectors, axis=0)

    return vec


def createVector(data, column, n_dim=300, min_count=3, sg=0, sample=0.001, window=5, hs=0, negative=5,
                 cbow_mean=1, iter=5, matrix=False):
    # Split text
    splitted = []
    lens = []
    for text in data[column]:
        split = text.split()
        splitted.append(split)
        lens.append(len(split))

    # Avg len
    avg_len = int(np.ceil(np.mean(lens)))

    # Initialize model and build vocab
    w2v = Word2Vec(splitted, size=n_dim, min_count=min_count, sg=sg, seed=1, sample=sample, window=window, hs=hs,
                   negative=negative, cbow_mean=cbow_mean, iter=iter, workers=1)

    if matrix:
        output = scale([buildWordVector(z, n_dim, w2v, matrix, avg_len=avg_len) for z in splitted])
        output = output.reshape((len(splitted), avg_len + 1, n_dim))
    else:
        output = scale([buildWordVector(z, n_dim, w2v, matrix) for z in splitted])

    return output