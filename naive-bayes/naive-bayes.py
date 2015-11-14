#!/usr/bin/python

from __future__ import print_function, division

from collections import namedtuple, defaultdict
from math import log, log1p

Datum = namedtuple('Datum', ['label', 'features'])

Classification = namedtuple('Classification', ['predicted', 'true'])

class NaiveBayesClassifier(object):

	def __init__(self):
		self.corpus_count = 0
		self.label_count = defaultdict(lambda: 0)
		self.feature_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

	def labels(self):
		return sorted(self.label_count.keys())

	def train(self, data):
		for datum in data:
			self.corpus_count += 1
			self.label_count[datum.label] += 1
			for i, v in enumerate(datum.features):
				self.feature_count[datum.label][i][v] += 1

	def classify(self, data):
		for datum in data:
			yield Classification(self.classify_features(datum.features), datum.label)

	def classify_features(self, features):
		# Maximum a posteriori estimation: arg max(P(features|label) for each label)
		return max(self.labels(), key=lambda label: self.log_posterior(features, label))

	def log_prior(self, label):
		# P(label) = # label / # all labels in training set
		# log P(label) = log # label - log # all labels in training set
		return log(self.label_count[label]) - log(self.corpus_count)

	def log_posterior(self, features, label):
		# P(features|label) = P(label) * P(label|features) / P(label)
		# log P(features|label) = log P(label) + log P(label|features) - log P(label)
		# (we leave out the evidence value, P(label), since it is constant)
		return self.log_prior(label) + self.log_likelihood(label, features)

	def log_likelihood(self, label, features):
		# P(label|features) = product(P(label|feature value) for each feature)
		# log P(label|features) = sum(log P(label|feature value) for each feature)
		return sum(self.log_feature_likelihood(label, i, v) for i, v in enumerate(features))

	def log_feature_likelihood(self, label, i, v):
		# P(label|feature value) = # value / # all values for feature
		# log P(label|feature value) = log # value - log # all values for feature
		# (we use Laplace smoothing, adding 1 to the numerator and denominator,
		# to handle zero counts (particularly since log 0 = -infinity))
		return log1p(self.feature_count[label][i][v]) - log1p(self.label_count[label])

def digit_data(label_filename, feature_filename):
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
			yield Datum(label, features)

def main():
	# Train naive Bayes classifier on training data
	nbc = NaiveBayesClassifier()
	training_data = digit_data('traininglabels.txt', 'trainingimages.txt')
	nbc.train(training_data)
	assert len(nbc.labels()) == 10

	# Classify test data, output classifications, and build a confusion matrix
	test_data = digit_data('testlabels.txt', 'testimages.txt')
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
