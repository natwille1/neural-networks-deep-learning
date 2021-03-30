import numpy as np
import matplotlib.pyplot as plt

#%%
# Exercises 1
# Sigmoid neurons simulating perceptrons, part I
# Suppose we take all the weights and biases in a network of perceptrons,
# and multiply them by a positive constant, c>0. Show that the behaviour of
# the network doesn't change.

np.random.seed(123)

def perceptron(x, w, b):
    output = []
    weighted_inputs = np.dot(x, w) + b
    print("weighted inputs: ", weighted_inputs)
    for var in weighted_inputs:
        if var > 0:
            print("greater than zero: ", var)
            output.append(1)
        elif var <= 0:
            print("less than zero: ", var)
            output.append(0)
    return output


x = np.array([1,2,3])
w = np.random.randn(len(x))
b = np.random.randn(len(x))

output = perceptron(x, w, b)

print(output)

## Multiply by positive constant
c = 5
w2 = w*c
b2 = b*c

output2 = perceptron(x, w2, b2)
print(output2)

#%%
# Exercises 2
# Sigmoid neurons simulating perceptrons, part II
# Suppose we have the same setup as the last problem - a network of perceptrons.
# Suppose also that the overall input to the network of perceptrons has been chosen.
# We won't need the actual input value, we just need the input to have been fixed.
# Suppose the weights and biases are such that w⋅x+b≠0 for the input x to any particular
# perceptron in the network. Now replace all the perceptrons in the network by
# sigmoid neurons, and multiply the weights and biases by a positive constant c>0.
# Show that in the limit as c→∞ the behaviour of this network of sigmoid neurons is
# exactly the same as the network of perceptrons. How can this fail when w⋅x+b=0 for one of the perceptrons?


def sigmoid(x, w, b):
    z = np.dot(x, w) + b
    output = 1.0/(1.0 + np.exp(-z))
    return output

for c in range(0, 100, 10):
    w3 = w*c
    b3 = b*c
    sig_out = sigmoid(x, w3, b3)
    print("c ", c, "output ", sig_out)


# Output for sigmoid neuron will still return 1 (if thresholld = 0.5)
# when w.x + b = 0, whereas perceptron will output 0

#%%
# There is a way of determining the bitwise representation of a digit by adding
# an extra layer to the three-layer network above. The extra layer converts the
# output from the previous layer into a binary representation, as illustrated in
# the figure below. Find a set of weights and biases for the new output layer.
# Assume that the first 3 layers of neurons are such that the correct output in
# the third layer (i.e., the old output layer) has activation at least 0.99, and
# incorrect outputs have activation less than  0.01

# Example output digit == 1, therefore first neuron = 0.99 and rest of activations = 0.01

third_layer = np.array([0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
