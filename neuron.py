import numpy
import math

class Neuron:
    def __init__(self, weights=[0], bias=-1):
        self.biasInput = bias
        self.weights = weights
        self.threshold = 0
        return

    def g(self, instance):
        sum = self.biasInput * self.weights[0]
        for i in range(0, len(instance)):
            sum += self.weights[i+1] * instance[i]
        return int(sum > self.threshold)

class SigmoidNeuron:
    def __init__(self, num_inputs = 1):
        # Weights will be created at time of initialization (including bias)
        # Function someone mentioned in class :)
        self.weights = numpy.random.ranf(num_inputs + 1) - .5
        return

    def g(self, input_array):
        # Bias calculation
        sum = -1 * self.weights[0]
        # Sum it with the rest of the inputs
        for i in range(len(input_array)):
            sum += self.weights[i + 1] * input_array[i]
        # This is where I use the sigmoid function
        return 1 / (1 + math.exp(-sum))