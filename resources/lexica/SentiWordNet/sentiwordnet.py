import pandas as pd
import os
import numpy as np
import sys


class sentiwordnet:
    def __init__(self):

        self.swn = self.load_swn()

    def load_swn(self):
        """
        Loads SentiWordnet to a dict
        :param filepath: file path of SentiWordNet
        :return:
        """

        filepath = "SentiWordNet_3.0.0_20130122.txt"
        swn_columns = ["POS", "ID", "PosScore", "NegScore", "SynsetTerms", "Gloss"]
        # Load SWN
        try:
            swn = pd.read_csv(filepath, sep="\t", skiprows=27, names=swn_columns)
        except:
            filepath = os.path.join(os.path.dirname(__file__), "SentiWordNet_3.0.0_20130122.txt")
            swn = pd.read_csv(filepath, sep="\t", skiprows=27, names=swn_columns)

        # Split synset terms and join them on swn
        swn = swn[swn["POS"].str.match("#") == False].reset_index(drop=True)
        synset_terms = swn["SynsetTerms"].str.split(expand=True)
        swn = pd.merge(swn, synset_terms, left_index=True, right_index=True)

        # Write synset terms from columns to rows
        swn = pd.melt(swn, id_vars=swn_columns, var_name="to_delete", value_name="SynsetTermWithNumber")
        swn = swn.dropna()

        # Split numbers of synset terms and join on swn
        synset_term_numbers = swn["SynsetTermWithNumber"].str.split("#", expand=True)
        synset_term_numbers.columns = ["SynsetTerm", "Number"]
        swn = pd.merge(swn, synset_term_numbers, left_index=True, right_index=True)

        # Make columns numeric
        swn["PosScore"] = pd.to_numeric(swn["PosScore"])
        swn["NegScore"] = pd.to_numeric(swn["NegScore"])
        swn["Number"] = pd.to_numeric(swn["Number"])

        # Rename POS Tags
        swn["POS"] = swn["POS"].replace({"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"})

        return swn[["POS", "ID", "PosScore", "NegScore", "SynsetTerm", "Number", "Gloss"]]

    def calc_swn_score(self, dataframe, agg_method, group=True):
        """
        Creates columns PosScore and NegScore of Sentiwordnet
        :param data: Dataframe with columns tweet_key and tweet
        :param agg_method: first, mean
        :return: SentiWordNet scores for text
        """

        # Warning useless here
        pd.options.mode.chained_assignment = None
        dataframe["second_index"] = dataframe.index.values
        pd.options.mode.chained_assignment = 'warn'

        # Join dataframe and SWN on word and pos
        swn_score = pd.merge(dataframe, self.swn, left_on=["1_gram_lemma", "1_gram_pos"],
                             right_on=["SynsetTerm", "POS"],
                             how="left")

        # Split data where no entry in swn was found
        # Data where nothing was found: (SWN columns will be dropped)
        no_swn_score = swn_score[swn_score.isnull().any(axis=1) == True]
        no_swn_score = no_swn_score.dropna(axis=1, how='all')

        # Columns where something was found:
        swn_score = swn_score[swn_score.isnull().any(axis=1) == False]

        # Where no swn entry found, join only on lemma_str
        no_swn_score = pd.merge(no_swn_score, self.swn, left_on=["1_gram_lemma"], right_on=["SynsetTerm"], how="left")

        # Union the two datasets
        swn_score = pd.concat([swn_score, no_swn_score])

        # Make numeric
        swn_score = swn_score.fillna(0)

        if group:
            if agg_method == "first":
                swn_score = swn_score[swn_score["Number"] <= 1]

            # This is done both when agg_method is mean or first
            swn_score = swn_score.groupby(["index"], as_index=False).agg({"PosScore": np.mean, "NegScore": np.mean})
        else:
            # Group back to the index
            swn_score = swn_score.groupby(["index", "1_gram_lemma", "1_gram_pos", "second_index"], as_index=False).agg(
                {"PosScore": np.mean, "NegScore": np.mean})

        swn_score["ObjScore"] = 1 - swn_score["PosScore"] - swn_score["NegScore"]

        return swn_score

    def lookup_word(self, word):

        score = self.swn.loc[self.swn["SynsetTerm"] == word, ["SynsetTerm", "PosScore", "NegScore"]]

        score = score.groupby(["SynsetTerm"], as_index=False).agg({"PosScore": np.mean, "NegScore": np.mean})
        score = score[["PosScore", "NegScore"]].values.tolist()

        if len(score) > 0:
            score = score[0]
            score.append(1)
            score = np.array(score)
        else:
            score = np.zeros(3)

        return score
