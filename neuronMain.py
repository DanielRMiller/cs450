import sys
import numpy as np
from copy import deepcopy
import random
def normalize(data):
    numAttributes = len(data[0])
    attributeValues = [[] for i in range(numAttributes)]
    for instance in data:
        for i, attribute in enumerate (instance):
            attributeValues[i].append(attribute)
    stndDev = np.std(attributeValues, axis=1)
    mean = np.mean(attributeValues, axis=1)
    for instance in data:
        for i, attribute in enumerate (instance):
            instance[i] = (attribute - mean[i]) / stndDev[i]
    return data
def readFile(fileName, data, targets):
    file = open(fileName, 'r')
    for line in file:
        array = line.split(',')
        size = len(array)
        array = [float(i) for i in array]
        data.append(array[0:(size - 1)])
        targets.append(array[size - 1])
    file.close()
    return
def randomize(data, targets):
    combine = list(zip(data, targets))
    random.shuffle(combine)
    data, targets = zip(*combine)
    return
class NeuralNetwork:
    def __init__(self, num, weights):
        self.nodes = [[] for i in range(num)]
        weights += 1
        for node in self.nodes:
            for i in range(weights):
                node.append([random.uniform(-1, 1)])
        self.weights = weights
    def train(self, trainData, targets):
        data = []
        for i, row in enumerate(trainData):
            data.append(np.append(row, -1))
        for i, row in enumerate(data):
            reshape = np.array(row).reshape(1, self.weights)
            matrix = np.dot(self.nodes, reshape)
            sum = np.sum(matrix.diagonal(0,1,2), 1)
            sum[sum > 0] = 1
            sum[sum <= 0] = 0
        return
def main(argv):
    inputs = {'-file': None, '-n': None, '-net': None}
    error = None
    for i, input in enumerate(argv):
        if input in inputs:
            if input == '-net':
                inputs[input] = True
            elif (i + 1) < len(argv):
                if input == '-file':
                    inputs[input] = argv[i + 1]
                elif input == '-n':
                    inputs[input] = int(argv[i + 1])
            else:
                error = ''
    if inputs['-n'] is None:
        error = ''
    if error is not None:
        print(error)
        return

    data = []
    targets = []
    if inputs['-file'] is not None:
        readFile(inputs['-file'], data, targets)
    else:
        from sklearn import datasets
        iris = datasets.load_iris();
        data = iris.data
        targets = iris.target
    data = normalize(data)
    randomize(data, targets)
    train = int(len(data) * 0.7)
    test  = len(data)
    trainSet     = data[0:train]
    testSet      = data[train:test]
    trainTargets = targets[0:train]
    testTargets  = targets[train:test]
    network = NeuralNetwork(inputs['-n'], len(data[0]))
    network.train(trainSet, trainTargets)
    return
if __name__ == '__main__':
    main(sys.argv)