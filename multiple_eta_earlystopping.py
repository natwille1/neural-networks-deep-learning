"""multiple_eta
~~~~~~~~~~~~~~~

This program shows how different values for the learning rate affect
training.  In particular, we'll plot out how the cost changes using
three different values for eta.

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2_L1_earlystopping as network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# Constants
EARLY_STOPPING = [10, 20, 30]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 60

def main():
    run_networks()
    make_plot()

def run_networks():
    """Train networks using three different values for the learning rate,
    and store the cost curves in the file ``multiple_eta.json``, where
    they can later be used by ``make_plot``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = []
    for eta in EARLY_STOPPING:
        print("\nTrain a network using eta = "+str(eta))
        net = network2.Network([784, 30, 10])
        epochs = NUM_EPOCHS * eta
        results.append(
            net.SGD(training_data[:1000], NUM_EPOCHS, 10, 0.5, lmbda=5.0,
                    evaluation_data=validation_data,
                    monitor_evaluation_accuracy=True, early_stopping_num=eta))
    f = open("multiple_nepocs_earlystopping.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_nepocs_earlystopping.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for eta, result, color in zip(EARLY_STOPPING, results, COLORS):
        _,evaluation_accuracy, _, _ = result
        ax.plot(np.arange(len(evaluation_accuracy)), evaluation_accuracy, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
    ax.set_xlim([0, len(evaluation_accuracy)])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    plt.legend(loc='upper right')
    plt.show()

#%%
import json
import matplotlib.pyplot as plt
import numpy as np

# Constants
EARLY_STOPPING = [10, 20, 30]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 60
f = open("/Users/nathalie.willems/Documents/AAAM/machine_learning/neural-networks-and-deep-learning/fig/multiple_nepocs_earlystopping.json", "r")
results = json.load(f)
f.close()
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111,)
for eta, result, color in zip(EARLY_STOPPING, results, COLORS):
    _, evaluation_accuracy,_, _ = result
    ax.plot(np.arange(len(evaluation_accuracy)), evaluation_accuracy, "o-",
                label="$\eta$ = "+str(eta),
                color=color)
ax.set_xlim([0, len(evaluation_accuracy)])
ax.set_xlabel('Epoch')
ax.set_ylabel('Evaluation Accuracy')
plt.legend(loc='upper right')
plt.show()


#%%

if __name__ == "__main__":
    main()
