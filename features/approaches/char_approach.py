import re
import sys
import numpy as np


def tokenizer(data, column):
    splitted = [list(z) for z in data[column]]

    max_len = max([len(chars) for chars in splitted])

    unique_items = np.unique(np.concatenate(splitted))

    matrix = []

    for doc in splitted:

        doc_vec = []

        for char in doc:

            char_vec = []

            for item in unique_items:
                if char == item:
                    char_vec.append(1)
                else:
                    char_vec.append(0)

            doc_vec.append(char_vec)

        fill_len = max_len - len(doc_vec)

        for i in range(fill_len):
            doc_vec.append(np.zeros(len(unique_items)))

        matrix.append(doc_vec)

    matrix = np.asarray(matrix)
    #matrix = matrix.reshape(len(matrix), max_len*len(unique_items))

    return matrix
