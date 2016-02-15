import random
import numpy
from neuron import SigmoidNeuron
class NetworkClassifier:
	def __init__(self, topology):
		# Recieved in init but used in train
		self.topology = topology
		# Built in train and used in predict to z-score the data
		self.std_devs = []
		self.means = []
		# This will be built in train and used in predict
		self.network = []
		# List of the set of targets (Output nodes should be in same order)
		self.target_list = []

	# Train: Accepts a training instances and targets. Does nothing with them.
	def train(self, instances, targets):
		self.target_list = list(set(targets))
		# Convert all strings to numbers
		instances = numpy.array([[float(string) for string in inner] for inner in instances])
		# Normalize Data
		for j in range(len(instances[0])):
			self.std_devs.append(numpy.std(instances[:, j]))
			self.means.append(numpy.mean(instances[:, j]))

			# Scale Data
			instances[:, j] -= self.means[j]
			instances[:, j] /= self.std_devs[j]

		# For each layer in our topology
		for i, layer in enumerate(self.topology):
			neurons = []
			if not i:
				# Create a node for each node in layer
				for j in range(layer):
					neurons.append(SigmoidNeuron(len(instances[0])))
			else:
				# Create a node for each node in layer
				for j in range(layer):
					neurons.append(SigmoidNeuron(len(self.network[i - 1])))
			# Add the neurons the the layer
			self.network.append(neurons)
		# Last layer will only have a node for each output
		neurons = []
		for j in range(len(set(targets))):
			neurons.append(SigmoidNeuron(len(self.network[-1])))
		self.network.append(neurons)
		return
	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		instances = numpy.array([[float(string) for string in inner] for inner in instances])
		for instance in instances:
			# scale the instance
			for i, val in enumerate(instance):
				instance[i] -= self.means[i]
				instance[i] /= self.std_devs[i]

			# outputs (We need to know who wins)
			outputs = []
			for i, layer in enumerate(self.network):
				layer_output = []
				if not i:
					for node in layer:
						layer_output.append(node.g(instance))
				else:
					for node in layer:
						layer_output.append(node.g(outputs))
				outputs = layer_output
			predictions.append(self.target_list[numpy.argmax(outputs)])
		print (predictions)
		return predictions;
	def fit(self, data, targets):
		self.train(data, targets)

