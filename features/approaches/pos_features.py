import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


def extract_pos_features(df_track1_words):
    # Group by index, pos and count
    word_count = df_track1_words.groupby(["index", "pos_numb"], as_index=False).agg({"1_gram_str": np.size})
    word_count = word_count.pivot_table("1_gram_str", 'index', 'pos_numb', fill_value=0)
    word_count.reset_index(inplace=True, drop=False)

    # Add missing values
    missing_indices = df_track1_words.loc[df_track1_words["pos_numb"].isnull(), ["index"]].drop_duplicates()
    word_count = pd.concat([word_count, missing_indices], ignore_index=True)

    # Sort by index, then drop index column
    word_count.sort_values(by="index", inplace=True)
    word_count.drop("index", axis=1, inplace=True)

    # Fill NaN values
    word_count.fillna(0, inplace=True)

    return scale(word_count.values)
