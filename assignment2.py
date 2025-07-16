# Rewrite the code specifically to meet your assignment requirements exactly:
# - Fully from scratch (no sklearn or any libraries for metrics)
# - Neural network with forward and backward propagation using gradient descent
# - Manual data loading and normalization
# - Manual MSE calculation and evaluation
# - Plot training loss
# - Output feature importance based on absolute weights

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Student_performance_data _.csv")

df = df.drop(columns=["StudentID", "GradeClass"])

features = df.drop(columns=["GPA"])
target = df["GPA"]

features_norm = (features - features.min()) / (features.max() - features.min())

X = features_norm.values
y = target.values.reshape(-1, 1)

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2 
        return self.output

    def backward(self, x, y):
        m = y.shape[0]
        error = self.output - y

        dW2 = np.dot(self.a1.T, error) / m
        db2 = np.sum(error, axis=0, keepdims=True) / m

        d_hidden = np.dot(error, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(x.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, x, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            self.forward(x)
            loss = np.mean((self.output - y) ** 2)
            self.backward(x, y)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        return losses


nn = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, learning_rate=0.1)
losses = nn.train(X_train, y_train, epochs=1000)

y_pred = nn.forward(X_test)
mse = np.mean((y_pred - y_test) ** 2)

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print("\\n--- Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.show()


feature_importance = np.abs(nn.W1).sum(axis=1)
feature_names = features.columns
importance = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

print("\\n--- Feature Importance ---")
for name, value in importance:
    print(f"{name}: {value:.4f}")