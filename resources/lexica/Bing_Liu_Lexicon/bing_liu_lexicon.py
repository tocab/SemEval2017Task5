import pandas as pd
import numpy as np
import sys

class bing_liu_lexicon:
    def __init__(self):

        self.bing_liu = self.load_lexicon()


    def load_lexicon(self):
        # Load data
        negative_words = "resources/lexica/Bing_Liu_Lexicon/negative-words.txt"
        positive_words = "resources/lexica/Bing_Liu_Lexicon/positive-words.txt"

        # Save data to dataframe and add score
        df_negative_words = pd.read_csv(negative_words, comment=";", encoding='latin-1', header=None, names=["word"])
        df_negative_words["bing_liu_score"] = -1

        df_positive_words = pd.read_csv(positive_words, comment=";", encoding='latin-1', header=None, names=["word"])
        df_positive_words["bing_liu_score"] = 1

        # Union both dataframes
        df_words = pd.concat([df_negative_words, df_positive_words], ignore_index=True)

        return df_words


    def calc_bing_liu_score(self, dataframe, word_column, group=True):
        """
        Calculate a score between -1 and 1 for every text message in a dataframe
        :param dataframe: dataframe with words
        :param word_column: word to be used (original word or lemma)
        :return: dataframe with column "score"
        """

        # Join words from lexicon on words of messages
        dataframe = pd.merge(dataframe, self.bing_liu, how="left", left_on=word_column, right_on="word")

        if group:
            # Group by index and aggregate score
            dataframe = dataframe.groupby("index", as_index=False).agg({"bing_liu_score": np.mean})

        # Fill NaN values with 0
        dataframe = dataframe.fillna(0)

        return dataframe


    def lookup_word(self, word):

        score = self.bing_liu.loc[self.bing_liu["word"] == word, "bing_liu_score"].values.tolist()

        if len(score) > 0:
            score.append(1)
            score = np.array(score)
        else:
            score = np.zeros(2)

        return score
