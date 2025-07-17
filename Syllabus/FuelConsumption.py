import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Load and preprocess dataset
dataset_path = keras.utils.get_file("auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.dropna()

# One-hot encode 'Origin'
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Train/test split
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize
train_stats = train_dataset.describe().transpose()
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Define model (simple 3-layer MLP)
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mse',
              metrics=['mae'])

# Train the model
model.fit(normed_train_data, train_labels, epochs=200, batch_size=32, verbose=1)

# Evaluate
loss, mae = model.evaluate(normed_test_data, test_labels, verbose=2)
print(f"\nTest MAE: {mae}")
