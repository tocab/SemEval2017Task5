from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os.path
import collections


def mean_model_prediction(dataset, features, train, test, track, realistic=False, to_csv=False):
    y = dataset["sentiment score"].values

    result = collections.OrderedDict()
    cos_dict = collections.OrderedDict()
    predicts = []
    result_df = dataset.ix[test]

    for feature in features:

        if features[feature] is None:
            continue

        feature_matrix = features[feature]

        if len(feature_matrix.shape) > 2:
            feature_matrix = feature_matrix.reshape(
                (feature_matrix.shape[0], feature_matrix.shape[1] * feature_matrix.shape[2]))

        clf = SVR()

        clf.fit(feature_matrix[train], y[train])
        # Predict
        predicted = clf.predict(feature_matrix)

        if not realistic:
            cos = cosine_similarity(predicted[test].reshape(1, -1), y[test].reshape(1, -1))
            cos_dict[feature] = cos[0][0]
            print("SVR_" + feature, cos, end=" ")
        result[feature] = predicted
        predicts.append(predicted)

    if "mean" in features:
        # Calculate mean predicted scores and get score from cosine
        meaned = np.mean(predicts, axis=0)
        if not realistic:
            cos = cosine_similarity(meaned[test].reshape(1, -1),
                                    dataset.loc[test, "sentiment score"].values.reshape(1, -1))
            cos_dict["mean"] = cos[0][0]
            print("SVR_mean", cos, end=" ")
        result["mean"] = meaned[test]
        result_df["sentiment score"] = meaned[test]

    if "median" in features:
        # Calculate mean predicted scores and get score from cosine
        meaned = np.median(predicts, axis=0)
        if not realistic:
            cos = cosine_similarity(meaned[test].reshape(1, -1),
                                    dataset.loc[test, "sentiment score"].values.reshape(1, -1))
            print("SVR_median", cos, end=" ")
        result["mean"] = meaned[test]

    if realistic:
        '''
        if track == 1:
            result_df[["id", "spans", "cashtag", "sentiment score"]].to_json("submission.json", orient="records")
        elif track == 2:
            result_df[["id", "text", "company", "sentiment score"]].to_json("submission.json", orient="records")
        '''
        return result_df

    if to_csv:
        dataset.loc[test, "prediction"] = result["mean"]

        i = 0
        while os.path.isfile("results_split_" + str(i) + ".csv"):
            i += 1

        dataset.to_csv("results_split_" + str(i) + ".csv")

    return result, cos_dict
