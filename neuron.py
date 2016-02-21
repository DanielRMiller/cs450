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
        # This will be used to update the weights. Should be set at time of error calc
        self.new_weights = numpy.random.ranf(num_inputs + 1) - .5
        # Errors should be held in Neuron and will be updated with every iteration
        self.error = 0
        self.output = 0
        return

    def g(self, input_array):
        # print('weights: ', self.weights)
        # print('input_array: ', input_array)
        # Bias calculation
        sum = -1 * self.weights[-1]
        # print('sum - bias calculation: ', sum)
        # Sum it with the rest of the inputs
        for i in range(len(input_array)):
            # print('summing at item i: ', i)
            sum += self.weights[i] * input_array[i]
            # print('weight: ', self.weights[i])
            # print('input: ', input_array[i])
            # print('sum: ', sum)
        # This is where I use the sigmoid function
        # print('final sum: ', sum)
        self.output = 1 / (1 + math.pow(math.e, -sum))
        # print('final output: ', self.output)
        return self.output