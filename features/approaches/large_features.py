import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import itertools
import nltk
from gensim.models.word2vec import Word2Vec
from features.approaches.lexicon import lexicon
from resources.utils import nlp_utils


def create_large_vector(data, column, train_ids):
    c_nlp_utils = nlp_utils.nlp_utils()
    data = c_nlp_utils.cleaning_for_spacy(data, column)

    # Get tweets as List for spacy
    texts = data[["index", column]].values.tolist()

    text_lens = [len(text) for text in data[column].str.split()]
    avg_text_len = int(np.ceil(np.mean(text_lens)))

    doc_pos = []
    words = []

    for index, tweet in texts:
        tokenized_tweet = tweet.split()
        possed = nltk.pos_tag(tokenized_tweet)
        possed = [x + (index,) for x in possed]

        if len(possed) == 0:
            possed = [("", "", index)]

        words = words + possed
        doc_pos.append([x[1] for x in possed])

    df = pd.DataFrame(words, columns=["word", "POS", "index"])
    lex_features = ["bing_liu_score", "maxdiff_score", "PosScore", "NegScore", "ObjScore", "vader_score",
                    "fin_lex_score"]
    # ,"1_gram_fin_lex_score"]
    lex = lexicon(lex_features, data, data.index if train_ids is None else train_ids)
    words_lex_features = lex.lexicon_features_per_word(df)

    unique_doc_pos = np.unique(np.concatenate(doc_pos))

    # vectors
    pos_one_hot = convert_one_hot(doc_pos, unique_doc_pos)
    w2v_gensim = createVector(data, column, matrix=True, n_dim=50)

    large_vec = concat_feature_vectors([words_lex_features,
                                        #pos_one_hot,
                                        w2v_gensim], avg_text_len)

    large_vec = large_vec.reshape([large_vec.shape[0], avg_text_len+1, int(large_vec.shape[1]/(avg_text_len+1))])

    return large_vec


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size, w2v, matrix, avg_len=0):
    vec = []
    for word in text:
        try:
            vec.append(w2v[word].reshape((1, size)))
        except KeyError:
            continue

    if matrix:
        if len(vec) > 0:
            vec = np.asarray(vec).reshape(len(vec), size)
        else:
            vec = np.zeros((1, size))

    else:
        vec = np.mean(vec, axis=0)
        vec = vec.reshape((1, size))

    return vec


def createVector(data, column, n_dim=50, min_count=1, sg=0, sample=0.001, window=5, hs=0, negative=5,
                 cbow_mean=1, iter=5, matrix=False):
    # Split text
    splitted = []
    lens = []
    for text in data[column]:
        split = text.split()
        splitted.append(split)
        lens.append(len(split))

    # Avg len
    avg_len = np.ceil(np.mean(lens))

    # Initialize model and build vocab
    w2v = Word2Vec(splitted, size=n_dim, min_count=min_count, sg=sg, seed=1, sample=sample, window=window, hs=hs,
                   negative=negative, cbow_mean=cbow_mean, iter=iter, workers=1)

    output = np.array([buildWordVector(z, n_dim, w2v, matrix, avg_len=avg_len) for z in splitted])

    return output


def convert_one_hot(vec, unique_items):
    lookup_matrix_pos = np.eye(len(unique_items))

    a = []
    for pos in vec:
        idx = np.array([np.where(unique_items == item) for item in pos])
        one_hot = lookup_matrix_pos[idx.flatten()]
        a.append(one_hot)

    return np.array(a)


def concat_feature_vectors(vector_list, new_vec_len):
    large_vec = []

    for i in range(vector_list[0].shape[0]):

        vector_tuple = ()
        for vector in vector_list:
            vector_tuple += (vector[i],)

        concat = np.concatenate(vector_tuple, axis=1)
        new_token_vector = []
        for cycle_count, cycle in enumerate(itertools.cycle(concat)):
            new_token_vector.append(cycle)
            if cycle_count == new_vec_len:
                break

        new_token_vector = np.asarray(new_token_vector)
        vec = new_token_vector.flatten()
        large_vec.append(vec)

    large_vec = scale(np.array(large_vec))

    return large_vec
