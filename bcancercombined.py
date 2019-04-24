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

total_runs = 1000
all_stats = np.empty((total_runs, 9))
sample = 0

while sample < total_runs:
    try:
        training_data = df.sample(train_count, replace=True)
        test_sample = df[~df.index.isin(training_data.index)]

        training_data = training_data.values
        training_labels = training_data[:, 0]
        training_data = np.delete(training_data, 0, 1)
        unlabeled_data = udf.values
        unlabeled_labels = unlabeled_data[:, 0]
        unlabeled_data = np.delete(unlabeled_data, 0, 1)

        print('cotraining training count: %s' % train_count)
        print('cotraining test count: %s' % test_count)
        print('cotraining unlabeled count: %s' % unlabeled_data.shape[0])
        print()

        cotrain_model = Cotrain()
        cotrain_model.initialize(training_data, training_labels)
        cotrain_model.fit(unlabeled_data, 0.75)

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

        cotrain_unpredicted_data = cotrain_model.get_unpredicted_data()
        kmeans = SemiSupervisedKMeans(num_clusters=2)
        kmeans.initialize(training_data, training_labels)
        kmeans.fit(unlabeled_data, 7)

        # Clustering prediction accuracy
        cluster_unlabeled_predictions = kmeans.get_unlabeled_predictions()
        cluster_unlabeled_predictions_set = set([tuple(x) for x in cluster_unlabeled_predictions])
        cluster_correct_matches = np.array([x for x in cluster_unlabeled_predictions_set & unlabeled_truth_set])
        cluster_label_acc = len(cluster_correct_matches) / len(cluster_unlabeled_predictions)
        print('clustering labeling accuracy: %s' % cluster_label_acc)
        print()

        labeled_data = np.insert(training_data, 0, training_labels, axis=1)
        remaining_unlabeled_predictions = kmeans.predict(cotrain_unpredicted_data, 6)
        complete_training_data = np.vstack([labeled_data, cotrain_unlabeled_predictions, remaining_unlabeled_predictions])

        # Remaining data labeling accuracy
        remaining_unlabeled_predictions_set = set([tuple(x) for x in remaining_unlabeled_predictions])
        remaining_preds_correct_matches = np.array([x for x in remaining_unlabeled_predictions_set & unlabeled_truth_set])
        remaining_preds_acc = len(remaining_preds_correct_matches) / len(remaining_unlabeled_predictions)
        print('clustering remaining data labeling accuracy: %s' % remaining_preds_acc)
        print()

        # Combined SVM performance
        labels_for_complete_training_data = complete_training_data[:, 0]
        complete_training_data_no_labels = complete_training_data[:, 1:]
        clf = SVC(gamma='auto', C=1)

        scores = cross_val_score(clf, complete_training_data_no_labels, labels_for_complete_training_data, cv=10)
        avg_score = scores.mean()
        print('combined svm cv scores: %s' % scores)
        print("combined svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print()

        total_unlabeled_count = len(unlabeled_data)
        cotrain_labels_given = len(cotrain_unlabeled_predictions)
        cluster_labels_given = len(cluster_unlabeled_predictions)
        remaining_pred_count = len(cotrain_unpredicted_data)
        remaining_pred_labels_given = len(remaining_unlabeled_predictions)
        run_stats = np.array([total_unlabeled_count,
                              cotrain_labels_given, cotrain_label_acc,
                              cluster_labels_given, cluster_label_acc,
                              remaining_pred_count, remaining_pred_labels_given, remaining_preds_acc,
                              avg_score])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
np.savetxt('bcancercombined.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)
