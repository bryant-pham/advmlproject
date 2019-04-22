import numpy as np
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


class Cotrain:
    def __init__(self):
        self.nn = None
        self.svm = None
        self.labeled_data = None
        self.labels = None
        self.enc = None
        self.num_labeled = 0

    def initialize(self, labeled_data, labels):
        self.labeled_data = labeled_data
        self.labels = labels
        num_unique_labels = len(np.unique(labels))

        nn_label_one_hot_encode = self.label_one_hot_encode(labels)
        shape = labeled_data[0].shape
        self.nn = Sequential()
        self.nn.add(Dense(32, activation='relu', input_shape=shape))
        # self.nn.add(Dense(64, activation='relu'))
        self.nn.add(Dense(units=num_unique_labels, activation='softmax'))
        self.nn.compile(optimizer='adadelta',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.nn.fit(labeled_data, nn_label_one_hot_encode, epochs=50, batch_size=10)

        self.svm = SVC(gamma='auto', C=1, probability=True)
        self.svm.fit(labeled_data, labels)
        # scores = cross_val_score(self.svm, labeled_data, labels, cv=10)
        # print('svm cv scores: %s' % scores)
        # print("svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def fit(self, unlabeled_data, conf_threshold):
        unlabeled_copy = np.copy(unlabeled_data)
        labels_for_unlabeled_data = np.full(unlabeled_copy.shape[0], -1)
        converged = False
        while not converged:
            nn_preds = self.nn.predict_proba(unlabeled_copy)
            nn_rows_and_preds_above_thresh = np.argwhere(nn_preds >= conf_threshold)

            svm_preds = self.svm.predict_proba(unlabeled_copy)
            svm_rows_and_preds_above_thresh = np.argwhere(svm_preds >= conf_threshold)

            nn_set = set([tuple(x) for x in nn_rows_and_preds_above_thresh])
            svm_set = set([tuple(x) for x in svm_rows_and_preds_above_thresh])
            matching_rows = np.array([x for x in nn_set & svm_set])
            converged = len(matching_rows) == 0
            if not converged:
                matching_rows = matching_rows[matching_rows[:, 0].argsort()]

                confident_row_indices = matching_rows[:, 0].flatten()
                confident_rows = unlabeled_copy[confident_row_indices]
                confident_labels = matching_rows[:, 1].flatten()
                self.labeled_data = np.vstack([self.labeled_data, confident_rows])
                self.labels = np.concatenate((self.labels, confident_labels))
                confident_one_hot_labels = self.label_one_hot_encode(confident_labels)
                converged = len(confident_labels) == 0
                unlabeled_copy = np.delete(unlabeled_copy, confident_row_indices, axis=0)
                self.num_labeled += len(confident_rows)
                labels_for_unlabeled_data[confident_row_indices] = confident_labels

                self.nn.fit(confident_rows, confident_one_hot_labels, epochs=50, batch_size=10)
                self.svm.fit(self.labeled_data, self.labels)

        print('total unlabeled: %s ' % len(unlabeled_data))
        print('total labels given: %s ' % self.num_labeled)

        return labels_for_unlabeled_data

    def label_one_hot_encode(self, labels):
        label_reshape = labels.reshape(-1, 1)
        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(label_reshape)
        return self.enc.transform(label_reshape).toarray()

    def get_full_pts(self):
        return np.insert(self.labeled_data, 0, self.labels, axis=1)  # Add label to unlabeled point
