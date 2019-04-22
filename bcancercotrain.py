import numpy as np
import pandas as pd
from cotrain import Cotrain
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

f = open('datasets/breastcancer/breastcancer-labeled.csv')
u = open('datasets/breastcancer/breastcancer-unlabeled.csv')
df = pd.read_csv(f)
udf = pd.read_csv(u)

total_count = df.shape[0]
train_count = int(total_count * 0.8)
test_count = total_count - train_count

training_data = df.sample(train_count)
df = df[~df.index.isin(training_data.index)]
test_sample = df[~df.index.isin(training_data.index)]

training_data = training_data.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
unlabeled_data = udf.values
unlabeled_labels = unlabeled_data[:, 0]
unlabeled_data = np.delete(unlabeled_data, 0, 1)

print('training count: %s' % train_count)
print('test count: %s' % test_count)
print('unlabeled count: %s' % unlabeled_data.shape[0])

cotrain_model = Cotrain()
cotrain_model.initialize(training_data, training_labels)
predicted_labels = cotrain_model.fit(unlabeled_data, 0.7)

# Unlabeled prediction accuracy
not_predicted = np.where(predicted_labels == -1)
predicted_labels = np.delete(predicted_labels, not_predicted)
unlabeled_labels = np.delete(unlabeled_labels, not_predicted)
print('labeling accuracy: %s' % (1 - np.mean(unlabeled_labels != predicted_labels)))

# SVM performance
all_data = cotrain_model.get_full_pts()
labels = all_data[:, 0]
all_data = all_data[:, 1:]
clf = SVC(gamma='auto', C=1)
# clf.fit(all_data, labels)

scores = cross_val_score(clf, all_data, labels, cv=10)
print('cv scores: %s' % scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
