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
		# For error rate purposes
		self.total = 0
		self.correct = 0
		self.learning_rate = .01
		self.epochs = 500

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
		# The actual Trainging occurs
		for epoch in range(self.epochs):
			# print('epoch: ', epoch)
			# restart our counts
			self.total = 0
			self.correct = 0
			for index, instance in enumerate(instances):
				# print('index: ', index)
				# print('instance: ', instance)
				self.backPropagate(instance, targets[index])
			print('Training Set: ', self.correct / self.total * 100)
		return

	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		instances = numpy.array([[float(string) for string in inner] for inner in instances])
		for instance in instances:
			predictions.append(self.target_list[numpy.argmax(self.feedForward(instance))])
		return predictions;

	def fit(self, data, targets):
		self.train(data, targets)
		return

	def feedForward(self, instance, scale=True):
		if scale:
			# scale the instance
			for i, val in enumerate(instance):
				instance[i] -= self.means[i]
				instance[i] /= self.std_devs[i]

		# outputs (We need to know who wins)
		outputs = []
		for layer_index, layer in enumerate(self.network):
			layer_output = []
			if not layer_index:
				for neuron_index, neuron in enumerate(layer):
					# print('For neuron ', layer_index, ',', neuron_index)
					# print('Activating on instance: ', instance)
					layer_output.append(neuron.g(instance))
			else:
				for neuron_index, neuron in enumerate(layer):
					# print('For neuron ', layer_index, ',', neuron_index)
					# print('Activating on instance: ', instance)
					layer_output.append(neuron.g(outputs))
			outputs = layer_output
		return outputs

	def backPropagate(self, instance, target):
		# print('instance: ', instance)
		if target == self.target_list[numpy.argmax(self.feedForward(instance, False))]:
			self.correct += 1
		self.total += 1
		# print('target: ', target)
		# print(self.target_list[numpy.argmax(self.feedForward(instance))])
		# print(self.feedForward(instance, False))
		# print('target: ', target)
		# Get all the errors (This can be optimized to update weights of next layer as we go)
		for layer_index, layer in reversed(list(enumerate(self.network))):
			# print('layer_index: ', layer_index)
			# print('layer', layer)
			if (layer_index == len(self.network) - 1):
				# OUTPUT LAYER
				for neuron_index, neuron in enumerate(layer):
					neuron_target = 1 if neuron_index == target else 0
					# print('neuron: ', neuron)
					# print('neuron weights: ', neuron.weights)
					# print('neuron output: ', neuron.output)
					# print('neuron target: ', neuron_target)
					# Compute Error
					neuron.error = neuron.output * (1 - neuron.output) * (neuron.output - neuron_target)
					# print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)
			else:
				for neuron_index, neuron in enumerate(layer):
					# print('neuron weights: ', neuron.weights)
					# print('neuron output: ', neuron.output)
					# Compute Error
					neuron.error = neuron.output * (1 - neuron.output) * self.sum_for_back(neuron_index, layer_index + 1)
					# print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)

		# Update Weights
		for layer_index, layer in enumerate(self.network):
			outputs = []
			# print('layer_index: ', layer_index)
			# print('layer', layer)
			# print('outputs: ', outputs)
			# outputs should look identical no matter where retrieved from
			if not layer_index:
				outputs = instance
				# print('layer ', layer_index - 1, ' outputs: ', outputs)
			else:
				for neuron_index, neuron in enumerate(self.network[layer_index - 1]):
					# print('Neuron ', layer_index, ',', neuron_index, 'output: ', neuron.output)
					outputs = numpy.append(outputs, numpy.array([neuron.output]))
					# print('layer ', layer_index - 1, ' outputs: ', outputs)
			# Add in the -1 bias because that is not in the input nor the nodes retrieved from
			outputs = numpy.append(outputs, numpy.array([-1]))
			# print('layer ', layer_index - 1, ' outputs: ', outputs)

			for neuron_index, neuron in enumerate(layer):
				# print('neuron ', layer_index, ',', neuron_index, ' output: ', neuron.output)
				# print('neuron ', layer_index, ',', neuron_index, ' error: ', neuron.error)
				# print('neuron ', layer_index, ',', neuron_index, ' weights: ', neuron.weights)
				for weight_index, weight in enumerate(neuron.weights):
					# print('weight_index: ', weight_index)
					# print('weight before update: ', weight)
					# print('weight before update: ', neuron.weights[weight_index])
					# print('.1 * neuron.error * outputs[weight_index]')
					# print(.1, ' * ', neuron.error, ' * ', outputs[weight_index])
					# print(.1 * neuron.error * outputs[weight_index])
					neuron.weights[weight_index] -= self.learning_rate * neuron.error * outputs[weight_index]
					# BAD -> 'weight after update: ', weight
					# print('weight after update: ', neuron.weights[weight_index])

	def sum_for_back(self, neuron_index_j, layer_index_k):
		sum = 0
		for neuron_k in self.network[layer_index_k]:
			# print('neuron_k.output: ', neuron_k.output)
			# print('neuron_k.error: ', neuron_k.error)
			# print('neuron_k.weights: ', neuron_k.weights)
			# print('neuron_k.weights[neuron_index_j]: ', neuron_k.weights[neuron_index_j])
			sum += neuron_k.error * neuron_k.weights[neuron_index_j]
			# print('sum after neuron: ', sum)
		# print('final sum: ', sum)
		return sum

