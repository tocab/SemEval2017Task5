import numpy as np


def sentiment_feature(dataset):
    dataset.loc[dataset["message.entities.sentiment.basic"].isnull(), "message.entities.sentiment.basic"] = "unknown"
    values = dataset["message.entities.sentiment.basic"].values
    unique_items = np.unique(values)

    one_hot_vector = convert_one_hot(values, unique_items)

    return one_hot_vector


def convert_one_hot(vec, unique_items):
    lookup_matrix_pos = np.eye(len(unique_items))

    a = []
    for item in vec:
        idx = np.where(unique_items == item)
        one_hot = lookup_matrix_pos[idx]
        a.append(one_hot.flatten())

    return np.array(a)
