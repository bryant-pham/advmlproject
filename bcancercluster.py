import numpy as np
import pandas as pd
from clustering import SemiSupervisedKMeans
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
print('training count: %s' % train_count)
print('test count: %s' % test_count)

training_data = training_data.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
unlabeled_data = udf.values
unlabeled_labels = unlabeled_data[:, 0]
unlabeled_data = np.delete(unlabeled_data, 0, 1)

kmeans = SemiSupervisedKMeans(num_clusters=2)
kmeans.initialize(training_data, training_labels)
kmeans.fit(unlabeled_data, 7)

# Label prediction accuracy setup
unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

# Clustering prediction accuracy
cluster_predicted_data = kmeans.get_unlabeled_predictions()
cluster_predicted_set = set([tuple(x) for x in cluster_predicted_data])
cluster_correct_matches = np.array([x for x in cluster_predicted_set & unlabeled_truth_set])
print('clustering labeling accuracy: %s' % (len(cluster_correct_matches) / len(cluster_predicted_data)))
print()

# Combined SVM performance
new_labeled_data = kmeans.get_full_labeled_data()
labels = new_labeled_data[:, 0]
new_labeled_data = new_labeled_data[:, 1:]
clf = SVC(gamma='auto', C=1)

scores = cross_val_score(clf, new_labeled_data, labels, cv=10)
print('cluster svm cv scores: %s' % scores)
print("cluster svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
