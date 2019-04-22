import numpy as np
import pandas as pd
from clustering import SemiSupervisedKMeans

f = open('datasets/breastcancer/breastcancer-labeled.csv')
u = open('datasets/breastcancer/breastcancer-unlabeled.csv')
df = pd.read_csv(f)
udf = pd.read_csv(u)
training_data = df.values
labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
unlabeled_data = udf.values
unlabeled_labels = unlabeled_data[:, 0]
unlabeled_data = np.delete(unlabeled_data, 0, 1)

kmeans = SemiSupervisedKMeans(num_clusters=2)
# training_data = np.array([[0, 1],[100, 100]])
# labels = np.array([0, 1])
# unlabeled_data = np.array([[0, 0, 25], [0, 100, 125], [0, 0, 1], [0, 100, 101]])
kmeans.initialize(training_data, labels)
# kmeans.fit(unlabeled_data, 3)
predicted_labels = kmeans.fit(unlabeled_data, 7)
print(1 - np.mean(unlabeled_labels != predicted_labels))