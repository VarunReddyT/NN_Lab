import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

# Load IMDb dataset, limit the vocabulary size to the most frequent 10,000 words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to a fixed length (e.g., 250)
max_sequence_length = 250
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))
model_lstm.add(LSTM(64)) # You can replace LSTM with GRU
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

loss, accuracy = model_lstm.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")