import numpy as np
import pandas as pd
from cotrain import Cotrain
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

f = open('datasets/breastcancer/breastcancer-labeled2.csv')
u = open('datasets/breastcancer/breastcancer-unlabeled2.csv')
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
        training_data = df.values
        training_labels = training_data[:, 0]
        training_data = np.delete(training_data, 0, 1)
        unlabeled_data = udf.values
        unlabeled_labels = unlabeled_data[:, 0]
        unlabeled_data = np.delete(unlabeled_data, 0, 1)

        print('training count: %s' % train_count)
        print('unlabeled count: %s' % unlabeled_data.shape[0])

        cotrain_model = Cotrain()
        cotrain_model.initialize(training_data, training_labels)
        cotrain_model.fit(unlabeled_data, 0.80)

        # Label prediction accuracy setup
        unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
        unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

        # Cotraining label prediction accuracy
        cotrain_unlabeled_predictions = cotrain_model.get_unlabeled_predictions()
        cotrain_unlabeled_predictions_set = set([tuple(x) for x in cotrain_unlabeled_predictions])
        cotrain_correct_matches = np.array([x for x in cotrain_unlabeled_predictions_set & unlabeled_truth_set])
        cotrain_label_acc = len(cotrain_correct_matches) / len(cotrain_unlabeled_predictions)
        print('cotraining labeling accuracy: %s' % cotrain_label_acc)
        print()

        # SVM performance
        new_labeled_data = cotrain_model.get_full_labeled_data()
        labels = new_labeled_data[:, 0]
        new_labeled_data = new_labeled_data[:, 1:]
        clf = SVC(gamma=0.0001, C=1000, kernel='rbf')

        scores = cross_val_score(clf, new_labeled_data, labels, cv=10)
        avg_score = scores.mean()
        print('cotrain cv scores: %s' % scores)
        print("cotrain Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        total_unlabeled_count = len(unlabeled_data)
        cotrain_labels_given = len(cotrain_unlabeled_predictions)
        run_stats = np.array([total_unlabeled_count,
                              cotrain_labels_given, cotrain_label_acc,
                              avg_score])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
stddev = all_stats[:, 3:].flatten().std()
mean_stats = np.insert(mean_stats, 4, stddev)
np.savetxt('bcancercotrain.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)
