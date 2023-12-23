import numpy as np
import matplotlib.pyplot as plt
import mnist 
import random

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
        self.biases = np.zeros((d,1))
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
        # Forward Propagation
        self.forward_prop(inputs)

        # Calculate errors for the output layer
        errors = [2 * (self.layers[self.L].activations - targets) * sigmoid_derivative(self.layers[self.L].z_activations)]
        # Backpropagation for Output Layer
        
        self.layers[self.L].weights -= learning_rate * (errors[0] * sigmoid_derivative(self.layers[self.L].z_activations) @ self.layers[self.L-1].activations.T)
        self.layers[self.L].biases -= learning_rate * (errors[0] * sigmoid_derivative(self.layers[self.L].z_activations))

        # Backpropagation for Hidden Layers
        for k in range(self.L-1, 0, -1):
            error = self.layers[k+1].weights.T @ errors[self.L - k - 1] * sigmoid_derivative(self.layers[k].z_activations)
            errors.append(error)

            self.layers[k].weights -= learning_rate * (error * sigmoid_derivative(self.layers[k].z_activations) @ self.layers[k-1].activations.T)
            self.layers[k].biases -= learning_rate * (error * sigmoid_derivative(self.layers[k].z_activations))

        return c / len(targets)


# Load MNIST data
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Flatten and normalize images
train_images_flat = train_images.reshape((-1, 28 * 28)) / 255.0

# Create and initialize the neural network
nn = NeuralNetwork([28*28, 64, 32, 16, 10])

# Training parameters
learning_rate = 0.025
epochs = 25

# Training loop
for epoch in range(epochs):
    c = 0
    for i in range(len(train_images_flat)):
      # Forward propagation
      inputs = train_images_flat[i]
      targets = np.zeros(output_size)
      targets[train_labels[i]] = 1

      # Backward propagation
      nn.backward_prop(inputs, targets, learning_rate)

      if np.argmax(nn.layers[nn.L].activations) == train_labels[i]:
        c += 1

    print(f"epoch {epoch+1}. acc: {(c / train_images_flat.shape[0]) * 100}%")
    # Visualize progress (plot some predictions)
    sample_image_index = 0
    sample_image = train_images[sample_image_index]
    sample_input = train_images_flat[sample_image_index]
    prediction = nn.forward_prop(sample_input)

    plt.subplot(1, epochs, epoch + 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f'Epoch {epoch + 1}\nPredicted: {np.argmax(prediction)}')
    plt.axis('off')

plt.show()



            


