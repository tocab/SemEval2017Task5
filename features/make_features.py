from features.approaches import lexicon
from features.approaches import word2vec
from features.approaches import pos_features
from features.approaches import doc2vec
from features.approaches import sentiment_feature
from features.approaches import char_approach
from features.approaches import large_features

def make_features(dataset, nlp_utils, features, feature_column, train_ids):
    # Available features in this method
    my_features = ["w2v_spacy", "w2v_spacy_matrix", "lexicon", "w2v_gensim", "w2v_gensim_matrix", "pos", "d2v_gensim",
                   "sentiment", "mean", "median"]

    # Lexicon features to be used
    lex_features = ["bing_liu_score", "maxdiff_score", "PosScore", "NegScore", "ObjScore", "vader_score",
                    "fin_lex_score"]  # , "2_gram_fin_lex_score"]

    # feature dict for saving features
    feature_dict = dict()

    # If any of these features occures, start spacy tokenization
    if len({"lexicon", "w2v_spacy", "w2v_spacy_matrix", "pos"}.intersection(set(features))) > 0:
        df_tokenized, w2v_spacy, w2v_spacy_matrix = nlp_utils.tokenize_text_in_df(dataset[["index", feature_column]])

        if "w2v_spacy" in features:
            feature_dict["w2v_spacy"] = w2v_spacy
        if "w2v_spacy_matrix" in features:
            feature_dict["w2v_spacy_matrix"] = w2v_spacy_matrix
        if "lexicon" in features:
            lex = lexicon.lexicon(lex_features, dataset, train_ids)
            feature_dict["lexicon"] = lex.lexicon_features(dataset, df_tokenized, feature_column)
        if "pos" in features:
            feature_dict["pos"] = pos_features.extract_pos_features(df_tokenized)

    # Create features
    if "w2v_gensim" in features:
        feature_dict["w2v_gensim"] = word2vec.createVector(dataset, feature_column)
    if "w2v_gensim_matrix" in features:
        feature_dict["w2v_gensim_matrix"] = word2vec.createVector(dataset, feature_column, matrix=True)
    if "d2v_gensim" in features:
        feature_dict["d2v_gensim"] = doc2vec.createVector(dataset, feature_column)
    if "sentiment" in features:
        feature_dict["sentiment"] = sentiment_feature.sentiment_feature(dataset)
    if "character" in features:
        feature_dict["character"] = char_approach.tokenizer(dataset, feature_column)
    if "large_features" in features:
        feature_dict["large_features"] = large_features.create_large_vector(dataset, feature_column, train_ids)
    if "mean" in features:
        feature_dict["mean"] = None
    if "median" in features:
        feature_dict["median"] = None

    return feature_dict
