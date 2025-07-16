import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)*0.1
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)*0.1

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2


    def forward_pass(self, x):
        self.input = x
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.tanh(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.tanh(self.output_layer_input)
        return self.output
    
    def backward_pass(self, y_true):
        output_error = y_true - self.output
        output_delta = output_error * self.tanh_derivative(self.output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.tanh_derivative(self.hidden_layer_output)

        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(self.input.T, hidden_delta) * self.learning_rate

        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

        return np.mean(output_error ** 2)

X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
Y = np.array([[0], [0], [1], [1]])

nn = NeuralNetwork(2, 3, 1)
losses = []

for i in range(1000):
    output = nn.forward_pass(X)
    loss = nn.backward_pass(Y)
    losses.append(loss)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

plt.plot(losses)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

print("\nFinal Predictions:")
output = nn.forward_pass(X)
for i, pred in enumerate(output):
    print(f"Input: {X[i]} => Predicted: {pred[0]:.4f}, Rounded: {round(pred[0])}")
