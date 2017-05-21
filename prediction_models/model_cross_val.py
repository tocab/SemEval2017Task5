from prediction_models.models import mean_model
from prediction_models.models import rnn_model3
from sklearn.model_selection import KFold
from archiv.other.plotting import plot_results
import collections
from features.approaches import lexicon
from resources.utils import nlp_utils
from features.approaches import large_features


def load_individual_lex_features(df_track1, train_ids):
    lex_features = ["bing_liu_score", "maxdiff_score", "PosScore", "NegScore", "ObjScore", "vader_score",
                    "fin_lex_score"]
    lex = lexicon.lexicon(lex_features, df_track1, train_ids)
    c_nlp_utils = nlp_utils.nlp_utils()
    feature_column = "spans_cleaned" if "spans" in df_track1.columns else "text_cleaned"
    df_tokenized, w2v_spacy, w2v_spacy_matrix = c_nlp_utils.tokenize_text_in_df(df_track1[["index", feature_column]])
    features = lex.lexicon_features(df_track1, df_tokenized, feature_column)

    return features


def model_cross_val(df_track1, features, track, folds=5, plot=False):
    # Make cross validation
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    kf_splits = list(kf.split(df_track1))

    save_results = collections.OrderedDict()
    result_list = []

    for i, train_test in enumerate(kf_splits):
        train = train_test[0]
        test = train_test[1]

        ###### Override features where train set is needed ######
        features["lexicon"] = load_individual_lex_features(df_track1, train)
        features["large_features"] = large_features.create_large_vector(df_track1,
                                                                        "spans_cleaned" if "spans" in df_track1.columns else "text_cleaned",
                                                                        train_ids=train)

        print(str(i + 1) + ". fold:", end="")

        # Approaches
        result_list.append(("SVM", mean_model.mean_model_prediction(df_track1, features, train, test, track=track)))
        result_list.append(("RNN", rnn_model3.rnn_model_prediction3(df_track1, features, train, test, track=track)))

        print("")

    if plot:
        plot_results(result_list, title="Track " + str(track))


def print_scores(result_dict):
    for feature in result_dict:
        feature_dict = result_dict[feature]
        print(feature, feature_dict["score"], end=" ")


def fill_dict(dict2fill, estimator_name, features, preds):
    for i, feature in enumerate(features):
        dict2fill[estimator_name + "_" + feature] = preds[i]

    return dict2fill
