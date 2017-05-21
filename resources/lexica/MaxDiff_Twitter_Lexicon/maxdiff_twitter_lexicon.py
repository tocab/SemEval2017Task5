import pandas as pd
import numpy as np
import sys

class maxdiff_twitter_lexicon:
    def __init__(self):
        self.maxdiff_lexicon = self.load_lexicon()


    def load_lexicon(self):
        # Load data
        negative_words = "resources/lexica/MaxDiff_Twitter_Lexicon/Maxdiff-Twitter-Lexicon_-1to1.txt"

        # Save data to dataframe and add score
        df_words = pd.read_csv(negative_words, comment=";", encoding='latin-1', header=None,
                               names=["maxdiff_score", "word"], sep="\t")

        return df_words


    def calc_maxdiff_score(self, dataframe, word_column, group=True):
        """
        Calculate a score between -1 and 1 for every text message in a dataframe
        :param dataframe: dataframe with words
        :param word_column: word to be used (original word or lemma)
        :return: dataframe with column "score"
        """

        # Join words from lexicon on words of messages
        dataframe = pd.merge(dataframe, self.maxdiff_lexicon, how="left", left_on=word_column, right_on="word")

        if group:
            # Group by index and aggregate score
            dataframe = dataframe.groupby("index", as_index=False).agg({"maxdiff_score": np.mean})

        # Fill NaN values with 0
        dataframe = dataframe.fillna(0)

        return dataframe


    def lookup_word(self, word):

        score = self.maxdiff_lexicon.loc[self.maxdiff_lexicon["word"] == word, "maxdiff_score"].values.tolist()

        if len(score) > 0:
            score.append(1)
            score = np.array(score)
        else:
            score = np.zeros(2)

        return score
