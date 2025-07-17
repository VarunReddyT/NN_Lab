import numpy as np


# 🔹 Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 🔹 Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)


# 🔧 Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))

        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        # 🔹 Forward pass: input → hidden → output
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.output = sigmoid(
            self.output_input
        )  # Optional: apply sigmoid here too if needed
        return self.output

    def backward(self, x, y, output):
        # 🔹 Calculate output error
        output_error = y - output
        output_delta = (
            output_error  # No activation derivative applied here for simplicity
        )

        # 🔹 Backpropagate error to hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 🔄 Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)

        self.weights_input_hidden += x.T.dot(hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, x, y, epochs):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)


# 🧪 Run the network
if __name__ == "__main__":
    # XOR input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    y = np.array([[0], [1], [1], [0]])

    # 🔧 Create and train the neural network
    input_size = 2
    hidden_size = 4
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    epochs = 10000
    nn.train(X, y, epochs)

    # 📊 Test the trained network
    print("\nPredictions after training:")
    for i in range(len(X)):
        prediction = nn.forward(X[i].reshape(1, -1))
        print("Input:", X[i], "Actual:", y[i], "Predicted:", np.round(prediction, 3))
