import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import categorical_accuracy
from sklearn.model_selection import StratifiedKFold


def label_one_hot_encode(labels):
    label_reshape = labels.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(label_reshape)
    return enc.transform(label_reshape).toarray()


f = open('datasets/breastcancer/breastcancer-labeled.csv')
df = pd.read_csv(f)

total_count = df.shape[0]
train_count = int(total_count * 0.8)
val_count = int(total_count * 0.1)
test_count = total_count - train_count - val_count

total_runs = 1
all_stats = np.empty((total_runs, 3))
sample = 0

kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []

while sample < total_runs:
    try:
        training_data = df.sample(train_count, replace=True)
        test_sample = df[~df.index.isin(training_data.index)]
        # remaining_df = df[~df.index.isin(training_data.index)]
        # val_sample = remaining_df.sample(val_count)
        # test_sample = remaining_df[~remaining_df.index.isin(val_sample.index)]

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
        nn.fit(training_data, nn_label_one_hot_encode, validation_split=0.2, epochs=50, batch_size=10)

        val_sample = val_sample.values
        val_labels = val_sample[:, 0]
        val_labels_one_hot_encode = label_one_hot_encode(val_labels)
        val_acc_value = nn.evaluate(val_sample[:, 1:], val_labels_one_hot_encode)[2]

        test_sample = test_sample.values
        test_labels = test_sample[:, 0]
        test_labels_one_hot_encode = label_one_hot_encode(test_labels)
        test_acc_value = nn.evaluate(test_sample[:, 1:], test_labels_one_hot_encode)[2]

        run_stats = np.array([train_count,
                              val_acc_value,
                              test_acc_value])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
np.savetxt('bcancercombined.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)