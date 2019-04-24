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
train_count = int(total_count)
test_count = total_count - train_count

total_runs = 100
all_stats = np.empty((total_runs, 4))
sample = 0

while sample < total_runs:
    try:
        training_data = df.sample(train_count, replace=True)
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
        cluster_unlabeled_predictions = kmeans.get_unlabeled_predictions()
        cluster_unlabeled_predictions_set = set([tuple(x) for x in cluster_unlabeled_predictions])
        cluster_correct_matches = np.array([x for x in cluster_unlabeled_predictions_set & unlabeled_truth_set])
        cluster_label_acc = len(cluster_correct_matches) / len(cluster_unlabeled_predictions)
        print('clustering labeling accuracy: %s' % cluster_label_acc)
        print()

        # Combined SVM performance
        new_labeled_data = kmeans.get_full_labeled_data()
        labels = new_labeled_data[:, 0]
        new_labeled_data = new_labeled_data[:, 1:]
        clf = SVC(gamma='auto', C=1)

        scores = cross_val_score(clf, new_labeled_data, labels, cv=10)
        avg_score = scores.mean()
        print('cluster svm cv scores: %s' % scores)
        print("cluster svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        total_unlabeled_count = len(unlabeled_data)
        cluster_labels_given = len(cluster_unlabeled_predictions)
        run_stats = np.array([total_unlabeled_count,
                              cluster_labels_given, cluster_label_acc,
                              avg_score])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
np.savetxt('bcancercluster.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)
