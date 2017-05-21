import pandas as pd
import numpy as np
from resources.utils import nlp_utils


class financial_lex():
    text_part = "text_str"

    def __init__(self, data, train_ids):

        data = data.ix[train_ids]

        c_nlp_utils = nlp_utils.nlp_utils()

        text_part = "spans" if "spans" in data.columns else "text"
        data = data[["id", text_part, "sentiment score"]]
        data["fin_lex_score"] = data["sentiment score"]
        data["index"] = data.index
        data[text_part] = data[text_part].str.lower()

        data = c_nlp_utils.expand_contractions(data, text_part)

        # Remove Cashtags
        data[text_part] = data[text_part].replace(r'[$][A-Za-z]+', r' ', regex=True)

        words = self.tokenize(data, text_part)
        self.finance_lex = words[["word", "fin_lex_score"]]

    def calc_fin_lex_score(self, dataframe, word_column, group=True):

        # Join words from lexicon on words of messages
        dataframe = pd.merge(dataframe, self.finance_lex, how="left", left_on=word_column, right_on="word")

        if group:
            # Group by index and aggregate score
            dataframe = dataframe.groupby("index", as_index=False).agg({"fin_lex_score": np.mean})

        # Fill NaN values with 0
        dataframe = dataframe.fillna(0)

        return dataframe

    def html_decode(self, data, column):
        data[column] = data[column].str.replace('&#39;', "'")
        data[column] = data[column].str.replace('&quot;', '"')
        data[column] = data[column].str.replace('&gt;', '>')
        data[column] = data[column].str.replace('&lt;', '<')
        data[column] = data[column].str.replace('&amp;', '&')

        return data

    def tokenize(self, data, text_part):
        count_documents = len(data)

        words = []
        for id, tweet, score in data[["id", text_part, "fin_lex_score"]].values.tolist():
            tokenized_tweet = tweet.split()

            tokenized_tweet = [(id, x, score) for x in tokenized_tweet]
            words += tokenized_tweet

        finance_lex_words = pd.DataFrame(words, columns=["id", "word", "fin_lex_score"])

        finance_lex_words = self.clean_words(finance_lex_words)

        grouped_words = finance_lex_words.groupby("word", as_index=False).agg(
            {"fin_lex_score": np.mean, "id": np.size})

        grouped_words["rel_count"] = grouped_words["id"] / count_documents * 100
        grouped_words = grouped_words[(grouped_words["rel_count"] >= 0.1) & (grouped_words["rel_count"] <= 10)]

        grouped_words = grouped_words.sort_values(by="fin_lex_score", ascending=False)

        return grouped_words

    def clean_words(self, words):
        words["word"] = words["word"].str.replace('[_,.:"?!=@|-]', '')

        words = words[words["word"].str.len() > 1]

        return words
