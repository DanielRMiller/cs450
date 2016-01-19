class KNNClassifier:
	"""A class that is to give the outputs to what it would be given the k nearest neighbors average."""
	def __init__(self, k=1):
		self.std_devs = []
		self.means = []
		self.k = k
		self.data = ""
		self.targets = ""
	# Train: Accepts a training instances and targets.
	def train(self, instances, targets): return
	# Predict: Accepts the test instances and returns a hard-coded response.
	def predict(self, instances):
		predictions = [];
		for instance in instances:
			predictions.append(0)
		return predictions;
	# Fit: This is so we can compare it to the sklearn algorithm (make switching easy)
	def fit(self, data, targets):
	    self.train(data, targets)
