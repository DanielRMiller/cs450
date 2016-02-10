import random
import numpy
class NetworkClassifier:
	def __init__(self, num_nodes, num_weights):
		# New blank array for each node
		self.nodes = [[] for i in range(num_nodes)]
		# Add in a bias input of -1 (well, the weight for it)
		self.num_weights = num_weights + 1
		# Each node should have (num_weights) randomized weights attached
		for node in self.nodes:
			for i in range(self.num_weights):
				node.append([random.uniform(-1, 1)])
	# Train: Accepts a training instances and targets. Does nothing with them.
	def train(self, instances, targets):
		data = []
		for i, row in enumerate(instances):
			data.append(numpy.append(row, -1))
		for i, row in enumerate(data):
			reshape = numpy.array(row).reshape(1, self.num_weights)
			matrix = numpy.dot(self.nodes, reshape)
			sum = numpy.sum(matrix.diagonal(0,1,2), 1)
			sum[sum > 0] = 1
			sum[sum <= 0] = 0
		return
	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		for instance in instances:
			predictions.append(0)
		return predictions;
	def fit(self, data, targets):
		self.train(data, targets)

