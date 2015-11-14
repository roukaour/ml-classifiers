#!/usr/bin/python

from __future__ import print_function, division

from collections import namedtuple, defaultdict
from math import log, log1p

Instance = namedtuple('Instance', ['label', 'features'])

Classification = namedtuple('Classification', ['predicted', 'true'])

class NaiveBayesClassifier(object):

	def __init__(self):
		self.num_instances = 0
		self.label_num = defaultdict(lambda: 0)
		self.fvl_num = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

	def labels(self):
		return sorted(self.label_num.keys())

	def train(self, instances):
		for instance in instances:
			self.num_instances += 1
			self.label_num[instance.label] += 1
			for f, v in enumerate(instance.features):
				self.fvl_num[f][v][instance.label] += 1

	def classify(self, instances):
		for instance in instances:
			yield Classification(self.classify_features(instance.features), instance.label)

	def classify_features(self, features):
		# Maximum a posteriori estimation: arg max(for each label: P(features|label))
		return max(self.labels(), key=lambda label: self.log_posterior(features, label))

	def log_posterior(self, features, label):
		# P(label|features) = P(label) * P(features|label) / P(features)
		# log P(label|features) = log P(label) + log P(features|label) - log P(features)
		# (we leave out the evidence value, P(features), since it is constant)
		return self.log_prior(label) + self.log_likelihood(features, label)

	def log_prior(self, label):
		# P(label) = # instances with label / # instances
		# log P(label) = log # instances with label - log # instances
		return log(self.label_num[label]) - log(self.num_instances)

	def log_likelihood(self, features, label):
		# P(features|label) = product(for each feature: P(feature value|label))
		# log P(features|label) = sum(for each feature: log P(feature value|label))
		return sum(self.log_feature_likelihood(f, v, label) for f, v in enumerate(features))

	def log_feature_likelihood(self, feature, value, label):
		# P(feature value|label) = # instances with label where feature has value / # instances with label
		# log P(feature value|label) = log # instances with label where feature has value - log # instances with label
		# (we use Laplace smoothing, adding 1 to the numerator and denominator, to handle zero counts;
		# this avoids errors due to log(0) being undefined)
		return log1p(self.fvl_num[feature][value][label]) - log1p(self.label_num[label])

	def entropy(self, feature):
		# H(feature) = -sum(for each feature value: P(value) * log P(value))
		# P(value) = # feature with value / # feature
		# log P(value) - log # feature with value - log # feature
		H = 0
		for value in self.fvl_num[feature]:
			numerator = sum(self.fvl_num[feature][value].values())
			if numerator > 0:
				H += (numer / self.num_instances) * (log(numer) - log(self.num_instances))
		return H

def digit_instances(label_filename, feature_filename):
	with open(label_filename, 'r') as label_file, open(feature_filename, 'r') as feature_file:
		for label_line in label_file:
			label = int(label_line)
			feature_lines = []
			for _ in range(28):
				feature_line = feature_file.readline().rstrip('\n')
				feature_lines.append(feature_line)
			features = ''.join(feature_lines)
			assert len(features) == 28 * 28
			assert set(features).issubset({' ', '+', '#'})
			yield Instance(label, features)

def main():
	# Train naive Bayes classifier on training data
	nbc = NaiveBayesClassifier()
	training_data = digit_instances('traininglabels.txt', 'trainingimages.txt')
	nbc.train(training_data)
	assert len(nbc.labels()) == 10

	# Classify test data, output classifications, and build a confusion matrix
	test_data = digit_instances('testlabels.txt', 'testimages.txt')
	confusion_matrix = [[0] * 10 for _ in range(10)]
	with open('predictedlabels.txt', 'w') as prediction_file:
		for result in nbc.classify(test_data):
			confusion_matrix[result.predicted][result.true] += 1
			prediction_file.write('%d\n' % result.predicted)

	# Calculate row totals
	for row in confusion_matrix:
		row.append(sum(row))
	# Calculate column totals
	confusion_matrix.append([sum(zip(*confusion_matrix)[i]) for i in range(len(confusion_matrix[0]))])
	# Calculate precision scores
	for i, row in enumerate(confusion_matrix):
		row.append(row[i] / row[-1])
	# Calculate sensitivity scores
	confusion_matrix.append([confusion_matrix[i][i] / confusion_matrix[-1][i]
		for i in range(len(confusion_matrix[0]) - 1)])
	# Calculate number of correct predictions
	confusion_matrix[-1].append(sum(confusion_matrix[i][i] for i in range(len(nbc.labels()))))
	# Add row labels
	for i, label in enumerate(nbc.labels()):
		confusion_matrix[i].append(label)
	# Add column labels
	confusion_matrix.append(list(nbc.labels()))
	# Add row and column total labels
	confusion_matrix[-3].append('total')
	confusion_matrix[-1].append('total')
	# Add precision and sensitivity labels
	confusion_matrix[-1].append('precision')
	confusion_matrix[-2].append('sensitivity')
	# Add overall axis label
	confusion_matrix[-1].append('pred\\true')
	# Move labels to top and left, not bottom and right
	for row in confusion_matrix:
		row.insert(0, row.pop())
	confusion_matrix.insert(0, confusion_matrix.pop())

	# Output confusion matrix
	for row in confusion_matrix:
		print(*row, sep='\t')

if __name__ == '__main__':
	main()
