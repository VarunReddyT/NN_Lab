import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# 1. Dataset Parameters
# -------------------------
batch_size = 128
total_words = 10000        # Vocabulary size
max_review_len = 80        # Maximum sentence length
embedding_len = 100        # Embedding size
units = 64
epochs = 20

# -------------------------
# 2. Load and Preprocess IMDB Dataset
# -------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# -------------------------
# 3. Create TensorFlow Dataset
# -------------------------
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# -------------------------
# 4. Define Simple RNN Model
# -------------------------

class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.outlayer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.embedding(inputs)  # [b, 80] -> [b, 80, 100]
        state0, state1 = self.state0, self.state1
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.outlayer(out1, training)  # [b, 64] -> [b, 1]
        prob = tf.sigmoid(x)
        return prob


# -------------------------
# 5. Compile, Train & Evaluate
# -------------------------

model = MyRNN(units)

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

print("\nTraining started...\n")
model.fit(train_ds, epochs=epochs, validation_data=test_ds)

print("\nEvaluating on test data...")
model.evaluate(test_ds)