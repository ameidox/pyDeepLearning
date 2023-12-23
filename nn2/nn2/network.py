import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mean_squared_error(predicted, target):
    return (predicted - target) ** 2

class Layer:
    # h - layer before length, d - current layer length

    # Activations - R^d
    # Biases - R^d
    # Weights - R^d*h

    def __init__(self, h, d):
        self.weights = np.random.uniform(-0.5, 0.5, size=(d, h))
        self.biases = np.random.uniform(-0.1, 0.1, size=(d, 1))

        self.z_activations = None # activations before activation function for backprop 
        self.activations = None  

    def forward_step(self, a): 
        # Ensure a is a column vector (matrix of shape (d, 1))
        d = self.weights.shape[1]  # Number of columns in self.weights
        a = np.reshape(a, (d, 1))

        self.z_activations = np.dot(self.weights, a) + self.biases
        self.activations = sigmoid(self.z_activations)
        return self.activations



class NeuralNetwork:
    def __init__(self, topology):
        self.layers = []
        for i in range(len(topology)):
            if i == 0:
                continue

            layer = Layer(topology[i-1], topology[i])
            self.layers.append(layer)
        self.L = len(self.layers) - 1

    def forward_prop(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_step(inputs)
        return inputs

    def backward_prop(self, inputs, targets, learning_rate):
        targets = np.reshape(targets, (10, 1))
        inputs = np.reshape(inputs, (784, 1))
        # Forward Propagation
        self.forward_prop(inputs)

        # Calculate errors for the output layer
        error = 2 * (self.layers[self.L].activations - targets)
        # Backpropagation for Output Layer
        
        self.layers[self.L].weights -= learning_rate * (error * sigmoid_derivative(self.layers[self.L].z_activations) @ self.layers[self.L-1].activations.T)
        self.layers[self.L].biases -= learning_rate * (error * sigmoid_derivative(self.layers[self.L].z_activations))
        # Backpropagation for Hidden Layers
        for k in range(self.L-1, -1, -1):
            error = self.layers[k+1].weights.T @ error * sigmoid_derivative(self.layers[k].z_activations)
            previusLayerActivations =  self.layers[k-1].activations.T if k > 0 else inputs.T
            self.layers[k].weights -= learning_rate * (error @ previusLayerActivations)
            self.layers[k].biases -= learning_rate * (error)

