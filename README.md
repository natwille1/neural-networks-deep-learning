# Neural_Networks_Deep_Learning
This repository contains python scripts that implement various neural networks for classification of the MNIST data set. 
The scripts implement solutions to problems set out in the "Neural Networks and Deep Learning" ebook by Michael Nielsen. 
The book can be found here: http://neuralnetworksanddeeplearning.com/index.html.

Each python script implements a different version of a feedforward neural networks with different features:

network.py = feedforward neural network with random weight initialisation (gaussian distribution with mean=0 and standard deviation=1), quadratic cost function,  and no regularisation.

network2_L1.py = optimised version of network.py with better weight initialisation (gaussian distribution with mean=0 and standard deviation=1 over the square root of the number of weights connecting to the same neuron), entropy cost function, and L1 regularisation. 

network2_L1_early_stopping.py = a custom implementation of some early stopping rules to prevent overfitting and/or lengthy training

network2_L1_learning_schedule.py = an implementation of a learning schedule for the learning rate 
