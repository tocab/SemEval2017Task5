import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import scale
from spacy.en import English
import json
import itertools
import sys


class nlp_utils:
    def __init__(self):
        self.nlp = English()
        self.stopwords = self.load_stopwords()
        self.cList = self.load_contractions()

    def load_stopwords(self):
        with open("resources/lexica/nltk_stopwords/nltk_stopwords.txt") as f:
            stopwords = f.read().splitlines()
        return stopwords

    def process_text(self, data, column_to_clean):
        """
        Cleans the text and extracts single words
        :param data: Dataframe with column "text" and "index"
        :return: Dataframe with word-level information like
        """

        ccolumn = column_to_clean + "_cleaned"

        data[ccolumn] = data[column_to_clean].str.lower()
        data = self.expand_contractions(data, ccolumn)

        # delete newline
        data[ccolumn] = data[ccolumn].replace(r'\n', r' ', regex=True)
        # delete urls
        data[ccolumn] = data[ccolumn].replace(r'(http(?:s)*:\/\/(?:[^\r\n\s]*))|(?:(pic\.twitter\.com)\/[^\r\n\s]*)',
                                              r' ', regex=True)
        # Space between terminal sign and last word
        data[ccolumn] = data[ccolumn].replace(r'(\S+)([?!.]+)$', r'\1 \2', regex=True)
        # Remove Hashtags and @
        # data[ccolumn] = data[ccolumn].replace(r'[@#](\S+)', r' \1 ', regex=True)
        # space between symbols
        data[ccolumn] = data[ccolumn].replace(r'[,:/\\().]', r' ', regex=True)
        # Remove Cashtags
        # data[ccolumn] = data[ccolumn].replace(r'[$][A-Za-z]+', r' ', regex=True)
        # Normalize numbers
        # data[ccolumn] = data[ccolumn].replace(r'[-+$0-9.%]+', ' ', regex=True)
        # normalize whitespace
        data[ccolumn] = data[ccolumn].replace(r'\s+', r' ', regex=True)
        data[ccolumn] = data[ccolumn].str.strip()
        # Remove stop words
        # data[ccolumn] = data[ccolumn].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

        # More ideas for the future:
        # Split words on capitalized letters like BringTheCupHome -> Bring The Cup Home
        # data["text_cleaned"] = data["text_cleaned"].replace(r'\s([A-Za-z]{1,2}[a-z]+)([A-Z]{1,2}[a-z]+)', r' \1 \2 ', regex=True)
        # remove elongations eg. awesomeeeee -> awesome
        # data["text_cleaned"] = data["text_cleaned"].replace(r'(.)\1{2,}', r'\1', regex=True)
        # convert numbers
        # data["spans_cleaned"] = data["spans_cleaned"].replace(r'-[$0-9.%]+', ' 1000 ', regex=True)
        # data["spans_cleaned"] = data["spans_cleaned"].replace(r'[+$0-9.%]+', ' -999 ', regex=True)

        return data

    def cleaning_for_spacy(self, tweets, column):
        # times
        tweets[column].replace(r'(\d)+x', r' \1 times ', regex=True, inplace=True)

        # lower and greater
        tweets[column].replace(r'<', r' lower ', regex=True, inplace=True)
        tweets[column].replace(r'>', r' greater ', regex=True, inplace=True)

        # Normalize numbers
        tweets[column].replace(r'(?:\s+|^)[+]?[$]\d+(?:[,.]\d+)?[km]?(?:\s+|$|\.)', ' positive price ',
                               regex=True, inplace=True)
        tweets[column].replace(r'(?:\s+|^)[+]?\d+(?:[,.]\d+)?[km]?[$](?:\s+|$|\.)', ' positive price ',
                               regex=True, inplace=True)
        tweets[column].replace(r'(?:\s+|^)-[.]?\d+(?:[,.]\d+)?[km]?(?:\s+|$|\.)', ' negative number ',
                               regex=True, inplace=True)
        tweets[column].replace(r'(?:\s+|^)[+]?[.]?\d+(?:[,.]\d+)?[km]?(?:\s+|$|\.)',
                               ' positive number ',
                               regex=True, inplace=True)
        tweets[column].replace(r'(?:\s+|^)[+]?\d+(?:[,.]\d+)?%(?:\s+|$|\.)', ' positive percent ',
                               regex=True, inplace=True)
        tweets[column].replace(r'(?:\s+|^)[-]\d+(?:[,.]\d+)?%(?:\s+|$|\.)', ' negative percent ',
                               regex=True, inplace=True)

        # & -> and
        tweets[column].replace(r'&', r' and ', regex=True, inplace=True)

        # "show me the $$$" -> "money"
        tweets[column].replace(r'[$]{3,}', r' money ', regex=True, inplace=True)

        # Plus
        tweets[column].replace(r'\+', r' plus ', regex=True, inplace=True)

        # Many points
        tweets[column].replace(r'\.{2,}', r' waiting ', regex=True, inplace=True)

        # normalize whitespace
        tweets[column].replace(r'\s+', r' ', regex=True, inplace=True)
        tweets[column] = tweets[column].str.strip()

        # delete signs
        tweets[column].replace('[_,.:"?!=@|#%-]', ' ', regex=True, inplace=True)

        return tweets

    def tokenize_to_ngramm(self, data, n):
        ngramm_dict = dict()
        i = 0

        for row in data.values:
            index = row[0]
            tweet = row[1]

            nlp_tweet = self.nlp(tweet)

            for token in nlp_tweet:

                try:
                    token.nbor(n - 1)
                except IndexError:
                    continue

                ngramm_str = ""
                ngramm_lemma = ""
                ngramm_pos = ""
                # Test if token has n neighbors
                for j in range(0, n):
                    ngramm_str += token.nbor(j).orth_ + " "
                    ngramm_lemma += token.nbor(j).lemma_ + " "
                    ngramm_pos += token.nbor(j).pos_ + " "

                ngramm_dict[i] = {"index": index,
                                  str(n) + "_gram_str": ngramm_str.strip(),
                                  str(n) + "_gram_lemma": ngramm_lemma.strip(),
                                  str(n) + "_gram_pos": ngramm_pos.strip()}
                i += 1

        words = pd.DataFrame.from_dict(ngramm_dict, orient='index')
        data = pd.merge(data, words, on=["index"], how="left")

        return data

    def tokenize_text_in_df(self, data, repeat_text_in_matrix=True):
        data = self.cleaning_for_spacy(data, data.columns[1])

        re_digit = re.compile(r"^\d+$")  # Detects digits
        re_whitespace = re.compile(r"^\s+$")  # Detects white space of any length

        # Get tweets as List for spacy
        texts = data.values.tolist()

        word_dict = dict()
        dict_key = 0
        doc_vecs = []
        text_lens = [len(text) for text in data.ix[:, 1].str.split()]
        avg_text_len = np.ceil(np.mean(text_lens))

        spcy_vectors = []

        for a, index_tweet in enumerate(texts):

            index = index_tweet[0]
            tweet = index_tweet[1]

            nlp_tweet = self.nlp(tweet)

            word_position = 0

            w2v_vector = np.zeros(300)
            token_vectors = []

            for token in nlp_tweet:

                lemma = token.lemma_.strip()

                if re_digit.match(lemma):
                    continue
                if len(lemma) <= 1:
                    continue
                if re_whitespace.match(lemma):
                    continue

                if token.has_vector:
                    w2v_vector += token.vector
                    token_vectors.append(token.vector)
                else:
                    token_vectors.append(np.zeros(300))

                word_dict[dict_key] = {"index": index,
                                       "lemma_numb": token.lemma,
                                       "1_gram_lemma": token.lemma_,
                                       "text_numb": token.orth,
                                       "1_gram_str": token.orth_,
                                       "pos_numb": token.pos,
                                       "1_gram_pos": token.pos_,
                                       "dep_str": token.dep_,
                                       "word_position": word_position}
                dict_key += 1
                word_position += 1

            if len(nlp_tweet) > 0:
                w2v_vector /= len(nlp_tweet)

            spcy_vectors.append(w2v_vector.reshape((1, 300)))

            # If tweet had no words, token_vector is empty
            if len(token_vectors) == 0:
                token_vectors.append(np.zeros(300))

            token_vectors = np.array(token_vectors)

            if repeat_text_in_matrix:

                new_token_vector = []
                for cycle_count, cycle in enumerate(itertools.cycle(token_vectors)):
                    new_token_vector.append(cycle)
                    if cycle_count == avg_text_len:
                        break

                new_token_vector = np.asarray(new_token_vector)

                doc_vecs.append(new_token_vector)
            else:
                doc_vecs.append(token_vectors)

        doc_vecs = scale(np.asarray(doc_vecs).flatten())
        doc_vecs = doc_vecs.reshape((len(texts), int(avg_text_len) + 1, 300))

        spcy_vectors = np.concatenate(spcy_vectors)
        spcy_w2v_matrix = scale(spcy_vectors)

        words = pd.DataFrame.from_dict(word_dict, orient='index')
        data = pd.merge(data, words, on=["index"], how="left")

        return data, spcy_w2v_matrix, doc_vecs

    def load_contractions(self):
        # source: http://devpost.com/software/contraction-expander

        # Open contractions json file
        with open("resources/lexica/contractions_expanded/contractions_expanded.json") as f:
            data = f.read()

        # Read data
        cList = json.loads(data)

        return cList

    def expand_contractions(self, data, column):

        for contraction in self.cList:
            data[column] = data[column].str.replace(contraction, self.cList[contraction])

        return data
