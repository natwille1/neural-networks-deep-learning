# Neural_Networks_Deep_Learning
This repository contains python scripts that implement various neural networks for classification of the MNIST data set. 
The scripts implement solutions to problems set out in the "Neural Networks and Deep Learning" ebook by Michael Nielsen. 
The book can be found here: http://neuralnetworksanddeeplearning.com/index.html.

Each python script implements a different version of a feedforward neural networks with different features:

network.py = feedforward neural network with random weight initialisation, quadratic cost function,  and no regularisation.

network2_unreg.py = optimised version of network.py with better weight initialisation and entropy cost function. 

network2_L1.py = optimised version of network2_unreg.py with L1 regularisation. 

network2_L1_early_stopping.py = a custom implementation of some early stopping rules to prevent overfitting and/or lengthy training for the network2_L1.py script.

network2_L1_learning_schedule.py = an implementation of a learning schedule for the learning rate for the network2_L1.py script.
