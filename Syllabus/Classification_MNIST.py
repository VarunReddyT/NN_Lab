import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build a 3-layer network
model = keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)  # No activation here since we'll use from_logits=True
])

# Load data
(x, y), _ = keras.datasets.mnist.load_data()
x = tf.reshape(x, (-1, 28 * 28)) / 255.0  # Normalize input

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
model.fit(x, y, epochs=5, batch_size=64)
# Evaluate
test_loss, test_accuracy = model.evaluate(x, y)
print(f"Test accuracy: {test_accuracy:.4f}")