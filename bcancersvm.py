import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

f = open('datasets/breastcancer/breastcancer-labeled2.csv')
df = pd.read_csv(f)

total_count = df.shape[0]
train_count = int(total_count)
test_count = total_count - train_count

total_runs = 500
all_stats = np.empty((total_runs, 2))
sample = 0
while sample < total_runs:
    try:
        training_data = df.sample(train_count, replace=True)
        test_sample = df[~df.index.isin(training_data.index)]

        training_data = training_data.values
        training_labels = training_data[:, 0]
        training_data = np.delete(training_data, 0, 1)

        print('training count: %s' % train_count)
        print('test count: %s' % test_count)

        # SVM performance
        clf = SVC(gamma='auto', C=1)

        scores = cross_val_score(clf, training_data, training_labels, cv=10)
        avg_score = scores.mean()
        print('solo svm cv scores: %s' % scores)
        print("solo svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        run_stats = np.array([train_count,
                              avg_score])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')

mean_stats = np.mean(all_stats, axis=0)
stddev = all_stats[:, 1:].flatten().std()
mean_stats = np.insert(mean_stats, 2, stddev)
np.savetxt('bcancersvm.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)
