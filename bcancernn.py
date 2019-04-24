import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import categorical_accuracy


def label_one_hot_encode(labels):
    label_reshape = labels.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(label_reshape)
    return enc.transform(label_reshape).toarray()


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
nn_label_one_hot_encode = label_one_hot_encode(training_labels)

print('training count: %s' % train_count)
print('test count: %s' % test_count)

# SVM performance
nn = Sequential()
nn.add(Dense(5, activation='relu', input_shape=training_data[0].shape))
nn.add(Dense(units=2, activation='softmax'))
nn.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy', categorical_accuracy])
nn.fit(training_data, nn_label_one_hot_encode, epochs=50, batch_size=10)
