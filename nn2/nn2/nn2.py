from network import *

import matplotlib.pyplot as plt
import mnist 
from tqdm import tqdm
import random

# Load MNIST data
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Flatten and normalize images
train_images_flat = train_images.reshape((-1, 28 * 28)) / 255.0

hidden_topology_input = input("Topology: ")
hidden_topology = list(map(int, hidden_topology_input.split()))

topology = [28*28] + hidden_topology + [10]

# Create and initialize the neural network
nn = NeuralNetwork(topology)

# Training parameters
learning_rate = float(input("Learning Rate: "))
epochs = int(input("Epochs: "))

biases_vector = nn.layers[0].biases.flatten()

# Training loop
for epoch in range(epochs):
    c = 0
    for i in tqdm(range(len(train_images_flat)), desc="Processing", unit="iteration"):
      # Forward propagation
      inputs = train_images_flat[i]
      targets = np.zeros(10)
      targets[train_labels[i]] = 1

      # Backward propagation
      nn.backward_prop(inputs, targets, learning_rate)

      if np.argmax(nn.layers[nn.L].activations) == train_labels[i]:
        c += 1


  
    print(f"\nepoch {epoch+1}. acc: {c / train_images_flat.shape[0] * 100:.2f}%")

def save_network(topology, layers, filename):
    with open(filename, 'w') as file:
        # Save topology
        file.write(f"Topology: {topology}\n")

        file.write("Layers:\n")
        for i in range(len(layers)):
            file.write(f"Layer {i}:\n")
            
            file.write(f"Weights:\n{str(layers[i].weights)}\n")
            
            # Flatten biases to a vector before writing
            biases_vector = layers[i].biases.flatten()
            file.write(f"Biases:\n{str(biases_vector)}\n")
            
            file.write(f"\n")


biases_vector = nn.layers[0].biases.flatten()
path = input("Enter the path to save: ")
filename = input("Enter the filename to save: ")

f = path + filename + ".nn"

save_network(topology, nn.layers, f)



            


