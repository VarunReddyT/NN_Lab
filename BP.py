import numpy as np

class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.rand(1)
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, x):
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_hidden
        self.output = self.sigmoid(self.output_layer_input)
        
        if self.output[0] > 0.5:
            # print("Output is greater than 0.5, returning 1")
            return self.output
        else:
            # print("Output is less than or equal to 0.5, returning 0")
            return self.output
    
    def backward_pass(self, x, y):
        output_error = np.mean((y - self.output) ** 2)
        output_delta = (y - self.output) * (self.output * (1 - self.output))

        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * (self.hidden_layer_output * (1 - self.hidden_layer_output))

        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(x.T, hidden_layer_delta) * self.learning_rate
        self.bias_hidden += np.sum(output_delta, axis=0) * self.learning_rate
        
    
nn = NeuralNetwork(2, 3, 1)

for i in range(1000):
    print(nn.forward_pass(np.array([[0, 0], [1, 1], [1, 0], [0, 1]])))
    nn.backward_pass(np.array([[0, 0], [1, 1], [1, 0], [0, 1]]), np.array([[0], [1], [1], [0]]))