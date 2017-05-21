from gensim.models import Doc2Vec
from sklearn.preprocessing import scale
from gensim.models.doc2vec import TaggedDocument


def createVector(data, column):
    # Split text
    splitted = [z.split() for z in data[column]]

    taggedDocs_single = []
    for i, sentence in enumerate(splitted):

        taggedDocs_single.append(TaggedDocument(sentence, [i]))

    d2v_single = Doc2Vec(taggedDocs_single, iter=1000)

    single_vectors = scale([doc for doc in d2v_single.docvecs])

    return single_vectors
