import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

f = open('datasets/breastcancer/breastcancer-labeled.csv')
df = pd.read_csv(f)

total_count = df.shape[0]
train_count = int(total_count * 0.8)
test_count = total_count - train_count

training_data = df.sample(train_count)
df = df[~df.index.isin(training_data.index)]
test_sample = df[~df.index.isin(training_data.index)]

training_data = training_data.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)

print('training count: %s' % train_count)
print('test count: %s' % test_count)

# SVM performance
clf = SVC(gamma='auto', C=1)
# clf.fit(all_data, labels)

scores = cross_val_score(clf, training_data, training_labels, cv=10)
print('solo svm cv scores: %s' % scores)
print("solo svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
