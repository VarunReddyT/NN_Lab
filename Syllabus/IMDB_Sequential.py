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
# 4. Define Sequential RNN Model
# -------------------------
model = keras.Sequential([
    layers.Embedding(input_dim=total_words, output_dim=embedding_len, input_length=max_review_len),
    layers.SimpleRNN(units, return_sequences=True, dropout=0.5),
    layers.SimpleRNN(units, dropout=0.5),
    layers.Dense(1, activation='sigmoid')
])

# -------------------------
# 5. Compile, Train & Evaluate
# -------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nTraining started...\n")
model.fit(train_ds, epochs=epochs, validation_data=test_ds)

print("\nEvaluating on test data...")
model.evaluate(test_ds)
