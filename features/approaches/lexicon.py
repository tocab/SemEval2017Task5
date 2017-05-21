import numpy as np
import pandas as pd
from resources.lexica.Bing_Liu_Lexicon import bing_liu_lexicon
from resources.lexica.MaxDiff_Twitter_Lexicon import maxdiff_twitter_lexicon
from resources.lexica.SentiWordNet import sentiwordnet
from resources.lexica.vaderLexicon import vader_lexicon
from sklearn.preprocessing import scale
from resources.lexica.financialLexicon import fin_lex_semeval


class lexicon:
    def __init__(self, features, data, train_ids):

        self.lexica = dict()
        self.lexica_names = []
        self.features = features

        for feature in features:
            if feature == "bing_liu_score":
                self.lexica_names.append("bing_liu")
            if feature == "maxdiff_score":
                self.lexica_names.append("maxdiff")
            if feature in ["PosScore", "NegScore", "ObjScore"]:
                self.lexica_names.append("swn")
            if feature == "vader_score":
                self.lexica_names.append("vader")
            if feature in ["fin_lex_score", "2_gram_fin_lex_score"]:
                self.lexica_names.append("fin")

        self.lexica_names = np.unique(self.lexica_names)

        for lex in self.lexica_names:
            if lex == "bing_liu":
                self.lexica[lex] = bing_liu_lexicon.bing_liu_lexicon()
            elif lex == "maxdiff":
                self.lexica[lex] = maxdiff_twitter_lexicon.maxdiff_twitter_lexicon()
            elif lex == "swn":
                self.lexica[lex] = sentiwordnet.sentiwordnet()
            elif lex == "vader":
                self.lexica[lex] = vader_lexicon.vader_lexicon()
            elif lex == "fin":
                self.lexica[lex] = fin_lex_semeval.financial_lex(data, train_ids)

    def lexicon_features(self, df_track1, df_track1_words, text_column):
        """
        Searches for lexicon features in text
        :param df_track1: dataframe with column index and text
        :return: dataframe with lexicon features
        """

        for lex in self.lexica_names:
            if lex == "bing_liu":
                # Calculate score from bing liu lexicon
                bing_liu_scores = self.lexica[lex].calc_bing_liu_score(df_track1_words[["index", "1_gram_lemma"]],
                                                                       "1_gram_lemma")
                df_track1 = pd.merge(df_track1, bing_liu_scores, on="index")
            elif lex == "maxdiff":
                # Calculate score from MaxDiff Twitter Lexicon
                maxdiff_scores = self.lexica[lex].calc_maxdiff_score(df_track1_words[["index", "1_gram_lemma"]],
                                                                     "1_gram_lemma")
                df_track1 = pd.merge(df_track1, maxdiff_scores, on="index")
            elif lex == "swn":
                # Calculate score from SentiWordnet
                swn_scores = self.lexica[lex].calc_swn_score(df_track1_words[["index", "1_gram_lemma", "1_gram_pos"]],
                                                             "first")
                df_track1 = pd.merge(df_track1, swn_scores, on="index")
            elif lex == "vader":
                # Calculate score from Vader Lexicon
                vader_scores = self.lexica[lex].calc_vader_score(df_track1_words[["index", "1_gram_lemma"]])
                df_track1 = pd.merge(df_track1, vader_scores, on="index")

                # Transform Vader scores from intervall -4 to 4 to intervall -1 to 1
                df_track1["vader_score"] = (((df_track1["vader_score"] - (-4)) * (1 - (-1))) / (4 - (-4))) + (-1)

            elif lex == "fin":
                fin_lex_scores = self.lexica[lex].calc_fin_lex_score(df_track1_words, "1_gram_lemma")
                df_track1 = pd.merge(df_track1, fin_lex_scores, on="index")

        # Scale
        df_track1[self.features] = scale(df_track1[self.features])

        return df_track1[self.features].values

    def lexicon_features_per_word(self, df_track1_words):
        """
        Searches for lexicon features in text
        :param df_track1: dataframe with column index and text
        :return: dataframe with lexicon features
        """

        words = df_track1_words.copy()

        # Add columns so that lexicon approaches work
        df_track1_words["1_gram_lemma"] = df_track1_words["word"]
        df_track1_words["1_gram_pos"] = df_track1_words["POS"]
        df_track1_words["1_gram_str"] = df_track1_words["word"]

        for lex in self.lexica_names:
            if lex == "bing_liu":
                # Calculate score from bing liu lexicon
                bing_liu_scores = self.lexica[lex].calc_bing_liu_score(df_track1_words[["index", "word"]],
                                                                       "word", group=False)

                words["bing_liu_score"] = bing_liu_scores["bing_liu_score"]

            elif lex == "maxdiff":
                # Calculate score from MaxDiff Twitter Lexicon
                maxdiff_scores = self.lexica[lex].calc_maxdiff_score(df_track1_words[["index", "word"]],
                                                                     "word", group=False)

                words["maxdiff_score"] = maxdiff_scores["maxdiff_score"]

            elif lex == "swn":
                # Calculate score from SentiWordnet
                swn_scores = self.lexica[lex].calc_swn_score(df_track1_words[["index", "1_gram_lemma", "1_gram_pos"]],
                                                             "first", group=False)

                words["PosScore"] = swn_scores["PosScore"]
                words["NegScore"] = swn_scores["NegScore"]
                words["ObjScore"] = swn_scores["ObjScore"]

            elif lex == "vader":
                # Calculate score from Vader Lexicon
                vader_scores = self.lexica[lex].calc_vader_score(df_track1_words[["index", "1_gram_lemma"]],
                                                                 group=False)

                # Transform Vader scores from intervall -4 to 4 to intervall -1 to 1
                vader_scores["vader_score"] = (((vader_scores["vader_score"] - (-4)) * (1 - (-1))) / (4 - (-4))) + (-1)

                words["vader_score"] = vader_scores["vader_score"]

            elif lex == "fin":
                # Finance Lexicon 1-gramm features
                fin_lex_score = self.lexica[lex].calc_fin_lex_score(df_track1_words, "1_gram_lemma", group=False)

                words["fin_lex_score"] = fin_lex_score["fin_lex_score"]

        # Scale
        words[self.features] = scale(words[self.features])

        stacked_lines = []
        for index in words["index"].drop_duplicates().values:
            word_scores = words.loc[words["index"] == index, self.features].values
            stacked_lines.append(word_scores)

        return np.array(stacked_lines)
