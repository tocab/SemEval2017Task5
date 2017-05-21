import pandas as pd
import os
import sys
import numpy as np


class vader_lexicon:
    def __init__(self):
        self.vader = self.load_lexicon()

    def load_lexicon(self):
        filepath = "vader_sentiment_lexicon.txt"
        vader_columns = ["word", "vader_score", "std_der", "rates"]

        # Load Vader lexicon
        try:
            vader = pd.read_csv(filepath, sep="\t", names=vader_columns, encoding="ISO-8859-14")
        except:
            filepath = os.path.join(os.path.dirname(__file__), "vader_sentiment_lexicon.txt")
            vader = pd.read_csv(filepath, sep="\t", names=vader_columns, encoding="ISO-8859-14")

        return vader

    def calc_vader_score(self, dataframe, group=True):
        """
        Calculates scores from Vader lexicon
        :param dataframe: dataframe with words (pos_str or lemma_str)
        :return: dataframe with scores
        """

        # Join vader lexicon on lemma_str
        dataframe = pd.merge(dataframe, self.vader, left_on="1_gram_lemma", right_on="word", how="left")

        if group:
            dataframe = dataframe.groupby("index", as_index=False).agg({"vader_score": np.mean})

        dataframe = dataframe.fillna(0)

        return dataframe

    def lookup_word(self, word):

        score = self.vader.loc[self.vader["word"] == word, "vader_score"].values.tolist()

        if len(score) > 0:
            score.append(1)
            score = np.array(score)
        else:
            score = np.zeros(2)

        return score
