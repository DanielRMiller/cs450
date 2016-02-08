import numpy as np

class Neuron:
    def __init__(self, weights, bias=1):
        self.biasInput = bias
        self.weights = weights
        self.threshold = 0
        return

    def g(self, instance):
        sum = self.biasInput * self.weights[0]
        for i in range(0, len(instance)):
            sum += self.weights[i+1] * instance[i]

        print("Sum = ", sum)
        return int(sum > self.threshold)