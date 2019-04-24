import numpy as np
import pandas as pd
from cotrain import Cotrain
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from clustering import SemiSupervisedKMeans

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

print('cotraining training count: %s' % train_count)
print('cotrainingtest count: %s' % test_count)
print('cotraining unlabeled count: %s' % unlabeled_data.shape[0])
print()

cotrain_model = Cotrain()
cotrain_model.initialize(training_data, training_labels)
cotrain_model.fit(unlabeled_data, 0.75)

# Label prediction accuracy setup
unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

# Cotraining label prediction accuracy
cotrain_predicted_data = cotrain_model.get_predicted_data()
cotrain_predicted_set = set([tuple(x) for x in cotrain_predicted_data])
cotrain_correct_matches = np.array([x for x in cotrain_predicted_set & unlabeled_truth_set])
print('cotraining labeling accuracy: %s' % (len(cotrain_correct_matches) / len(cotrain_predicted_data)))
print()

cotrain_unpredicted_data = cotrain_model.get_unpredicted_data()
kmeans = SemiSupervisedKMeans(num_clusters=2)
kmeans.initialize(training_data, training_labels)
kmeans.fit(cotrain_unpredicted_data, 4)

# Clustering prediction accuracy
cluster_predicted_data = kmeans.get_predicted_data()
cluster_predicted_set = set([tuple(x) for x in cluster_predicted_data])
cluster_correct_matches = np.array([x for x in cluster_predicted_set & unlabeled_truth_set])
print('clustering labeling accuracy: %s' % (len(cluster_correct_matches) / len(cluster_predicted_data)))
print()

# Combined SVM performance
new_labeled_data = np.vstack([cotrain_model.get_full_labeled_data(), kmeans.get_full_labeled_data()])
labels = new_labeled_data[:, 0]
new_labeled_data = new_labeled_data[:, 1:]
clf = SVC(gamma='auto', C=1)
# clf.fit(all_data, labels)

scores = cross_val_score(clf, new_labeled_data, labels, cv=10)
print('combined svm cv scores: %s' % scores)
print("combined svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
