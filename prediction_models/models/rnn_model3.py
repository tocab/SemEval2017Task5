# Example from blog post at:
# http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import collections

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class rnn_model:
    def __init__(self, data, target, dropout, training, num_layers=2, layer_count=10):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_layers = num_layers
        self._training = training
        self._layer_count = layer_count
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # Recurrent network.
        #network = tf.contrib.rnn.BasicLSTMCell(self._layer_count)
        # network = tf.nn.rnn_cell.GRUCell(self._layer_count)
        # network = lstm_bn_cell.BNLSTMCell(self._layer_count, self._training)
        #network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)

        cells = []
        for _ in range(self._num_layers):
            cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self._layer_count), output_keep_prob=self.dropout)
            cells.append(cell)
        network = tf.contrib.rnn.MultiRNNCell(cells)

        #network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        # Get values from last state only (shape [50,5,5] -> [50,5]
        output = output[:, - 1, :]

        # Sum to [50,1]
        output = tf.reduce_sum(output, axis=1)

        return output

    @lazy_property
    def cost(self):
        mae = tf.reduce_mean(tf.abs(tf.subtract(self.prediction, self.target)))
        return mae

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        return optimizer

    @lazy_property
    def error(self):
        mistakes = tf.reduce_mean(tf.abs(tf.subtract(self.target, self.prediction)))
        return mistakes

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def rnn_model_prediction3(df_track1, features, train, test, track, realistic=False):
    result_dict = collections.OrderedDict()
    result_df = df_track1.ix[test]
    cos_dict = collections.OrderedDict()
    for feature in features:

        if features[feature] is None:
            continue

        # RNN model with best params after GS
        # subtask 1 params:
        batch_size = 187
        p_dropout = 0.51
        p_layer_count = 15
        p_num_layers = 8
        epochs = 1000

        # Best params for subtask 2
        if track == 2:
            batch_size = 95
            p_dropout = 0.5738
            p_layer_count = 18
            p_num_layers = 2
            epochs = 3333

        predict, score = run(features[feature], df_track1["sentiment score"], train, test, plot=False,
                             batch_size=batch_size, p_dropout=p_dropout, p_layer_count=p_layer_count,
                             p_num_layers=p_num_layers, epochs=epochs)
        result_dict[feature] = predict

        if not realistic:
            cos_dict[feature] = score[0][0]
            print("RNN3_" + feature, score, end=" ")

    if "mean" in features:

        predicts = []
        for feature in features:
            if features[feature] is None:
                continue

            predicts.append(result_dict[feature])

        predicts = np.array(predicts)

        # Calculate mean predicted scores and get score from cosine
        meaned = np.mean(predicts, axis=0)
        result_dict["mean"] = meaned
        result_df["sentiment score"] = meaned

        if not realistic:
            cos = cosine_similarity(meaned.reshape(1, -1), df_track1.loc[test, "sentiment score"].values.reshape(1, -1))
            cos_dict["mean"] = cos[0][0]
            print("RNN3_mean", cos, end=" ")

    if realistic:
        return result_df
        result_df[["id", "spans", "sentiment score"]].to_json("submission.json", orient="records")

    return result_dict, cos_dict


def run(X, y, train, test, plot=False, verbose=0, batch_size=50, early_stopping=False, p_dropout=0.8, p_num_layers=2,
        p_layer_count=10, epochs=1000):
    # Normalize input data (important for automatic grid search)
    batch_size = int(np.rint(batch_size))
    p_num_layers = int(np.rint(p_num_layers))
    p_layer_count = int(np.rint(p_layer_count))
    epochs = int(np.rint(epochs))

    if len(X[0].shape) > 1:
        height = X[0].shape[0]
        width = X[0].shape[1]
    else:
        height = 1
        width = X[0].shape[0]
        X = X.reshape((-1, 1, X[0].shape[0]))

    # Make test train split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    # Save y without indices
    y_test = y_test.values
    y_train = y_train.values

    data = tf.placeholder(tf.float32, [None, height, width])
    target = tf.placeholder(tf.float32)
    dropout = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)

    model = rnn_model(data, target, dropout, training, num_layers=p_num_layers, layer_count=p_layer_count)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_error = []
    test_error = []

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Test always on same batch
    random_idx2 = np.random.randint(X_test.shape[0], size=100)

    for epoch in range(epochs):
        # Create train_batch
        random_idx = np.random.randint(X_train.shape[0], size=batch_size)
        batch = X_train[random_idx]
        y_batch = y_train[random_idx]

        # Optimize with train batch
        sess.run(model.optimize, {
            data: batch, target: y_batch, dropout: p_dropout, training: True})

        # Calc error for train batch
        batch_error = sess.run(model.error, {
            data: batch, target: y_batch, dropout: 1, training: False})

        # Create test_batch
        # random_idx2 = np.random.randint(X_test.shape[0], size=batch_size)
        test_batch = X_test[random_idx2]
        test_y_batch = y_test[random_idx2]

        # Calc error for test batch
        test_batch_error = sess.run(model.error, {
            data: test_batch, target: test_y_batch, dropout: p_dropout, training: False})

        # Print errors
        if verbose == 1 and epoch % 100 == 0:
            print('Epoch', epoch, 'train error ', batch_error, 'test error', test_batch_error)

        # Save in list for plot
        train_error.append(batch_error)
        test_error.append(test_batch_error)

        if early_stopping and batch_error < 0.25:
            break

    # saver.save(sess, "saved_weights/rnn_char_100000.ckpt")

    if plot:
        # Prepare plot
        plt.title("RNN: Train vs. Test")
        plt.plot(train_error, label="train", linewidth=1.5)
        plt.plot(test_error, label="test", linewidth=1.5, color="red")
        plt.ylabel("MAE")
        plt.xlabel("Epochs")
        plt.legend(loc=4)

        # show plot
        plt.show()

    output = sess.run(model.prediction, {
        data: X_test, target: y_test, dropout: 1, training: False})

    cs = cosine_similarity(output.reshape(1, -1), y_test.reshape(1, -1))

    tf.reset_default_graph()

    return output, cs


def main():
    pass


if __name__ == '__main__':
    main()
