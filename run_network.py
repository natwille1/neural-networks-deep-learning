import mnist_loader
import network_cross_entropy as network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data[:1000], 30, 10, 0.5,
              test_data=test_data)
