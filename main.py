from datasets import load_dataset
from features import make_features
from resources.utils import nlp_utils
from prediction_models import model_cross_val
import pandas as pd
from prediction_models.models import mean_model
from prediction_models.models import rnn_model3
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="path and filename to train data")
    parser.add_argument("--test_file", help="path and filename to test data")
    parser.add_argument("--mode", help="cross validation or prediction")
    parser.add_argument("--subtask", help="number of subtask of SemEval-2017 Task 5", type=int)
    parser.add_argument("--regressor", help="If mode is predict, select regressor for prediction")
    args = parser.parse_args()

    if args.mode == "cv":
        training(args.subtask, args.train_file)

    if args.mode == "predict":
        realistic(args.subtask, args.regressor, args.train_file, args.test_file)


def realistic(track, regressor, train_file, test_file):
    # load nlp utils
    c_nlp_utils = nlp_utils.nlp_utils()

    # Load data
    if track == 1:
        dataset_train = load_dataset.load_track1_data(train_file)
        dataset_test = load_dataset.load_track1_test_data(test_file)
    else:
        dataset_train = load_dataset.load_track2_full_data(train_file)
        dataset_test = load_dataset.load_track2_test_data(test_file)

    dataset_train["data"] = "train"
    dataset_test["data"] = "test"
    len_test = len(dataset_test)

    # Merge datasets
    corresponding_columns = [x for x in dataset_train.columns if x in dataset_test.columns]
    dataset = pd.concat([dataset_train[corresponding_columns], dataset_test[corresponding_columns]], ignore_index=True)
    dataset["index"] = dataset.index

    # Clean text
    if "text" in dataset.columns:
        dataset = c_nlp_utils.process_text(dataset, "text")
        feature_column = "text_cleaned"
    if "spans" in dataset.columns:
        dataset = c_nlp_utils.process_text(dataset, "spans")
        feature_column = "spans_cleaned"

    train_ids = dataset[dataset["data"] == "train"].index
    test_ids = dataset[dataset["data"] == "test"].index

    # Make features
    features = make_features.make_features(dataset,
                                           c_nlp_utils,
                                           features=["w2v_spacy_matrix",
                                                     "w2v_gensim_matrix",
                                                     "w2v_spacy",
                                                     "lexicon",
                                                     "w2v_gensim",
                                                     "large_features",
                                                     "mean"],
                                           feature_column=feature_column,
                                           train_ids=train_ids)

    if regressor == "svm":
        result_df = mean_model.mean_model_prediction(dataset, features, train_ids, test_ids, track=track,
                                                     realistic=True)
    elif regressor == "rnn":
        result_df = rnn_model3.rnn_model_prediction3(dataset, features, train_ids, test_ids, track=track,
                                                     realistic=True)

    if track == 1:
        result_df[["id", "spans", "cashtag", "sentiment score"]].to_json("submission.json", orient="records")
    elif track == 2:
        result_df[["id", "text", "company", "sentiment score"]].to_json("submission.json", orient="records")

    if len(result_df) == len_test:
        print("Everything good!")
    else:
        print("be carefull!!!! some tweets may be lost. This makes the score worse.")


def training(track, train_file):
    c_nlp_utils = nlp_utils.nlp_utils()

    # Load dataset
    if track == 1:
        dataset = load_dataset.load_track1_data(train_file)
    else:
        dataset = load_dataset.load_track2_full_data(train_file)

    # Clean text
    if "text" in dataset.columns:
        dataset = c_nlp_utils.process_text(dataset, "text")
        feature_column = "text_cleaned"
    if "spans" in dataset.columns:
        dataset = c_nlp_utils.process_text(dataset, "spans")
        feature_column = "spans_cleaned"

    # Make features
    features = make_features.make_features(dataset,
                                           c_nlp_utils,
                                           features=["w2v_spacy_matrix",
                                                     "w2v_gensim_matrix",
                                                     "w2v_spacy",
                                                     "w2v_gensim",
                                                     "mean"
                                                     ],
                                           feature_column=feature_column,
                                           train_ids=dataset.index)

    # Evaluate models
    model_cross_val.model_cross_val(dataset, features, track, plot=True)


if __name__ == "__main__":
    main()
