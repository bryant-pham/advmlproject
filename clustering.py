import numpy as np
import sys


class SemiSupervisedKMeans:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.centers = None
        self.cluster_pts = None
        self.num_labeled = None
        self.dist_traveled = None

    def initialize(self, labeled_data, labels):
        num_attr = labeled_data.shape[1]
        classes = np.unique(labels)
        self.cluster_pts = dict()
        self.centers = np.zeros((self.num_clusters, num_attr))
        self.dist_traveled = np.zeros(self.num_clusters)
        self.num_labeled = np.zeros(self.num_clusters)
        data_with_labels = np.insert(labeled_data, 0, labels, axis=1)
        for i, label in enumerate(classes):
            data_matching_label = data_with_labels[data_with_labels[:, 0] == label]
            # remove class column
            data_remove_label = data_matching_label[:, 1:] 
            center = np.sum(data_remove_label, axis=0) / data_remove_label.shape[0]
            self.cluster_pts[label] = data_matching_label
            self.centers[i] = center

    def fit(self, unlabeled_data, threshold):
        labels_for_unlabeled_data = np.full(unlabeled_data.shape[0], -1)
        data_copy = np.copy(unlabeled_data)
        diff = 1
        while diff != 0:
            indicies_delete_from_unlabeled = list()
            for i, unlabeled in enumerate(data_copy):
                label = self.closest_cluster_index(unlabeled, threshold)
                if label > -1:
                    indicies_delete_from_unlabeled.append(i)
                    self.num_labeled[label] += 1
                    labels_for_unlabeled_data[i] = label
                    unlabeled = np.insert(unlabeled, 0, label) # Add label to unlabeled point
                    self.cluster_pts[label] = np.vstack([self.cluster_pts[label], unlabeled])
            self.recompute_centers()
            num_before_remove = data_copy.shape[0]
            data_copy = np.delete(data_copy, indicies_delete_from_unlabeled, axis=0)
            num_after_remove = data_copy.shape[0]
            diff = num_before_remove - num_after_remove
        print('c_score: %s' % self.c_score())
        print('dist_traveled: %s' % self.dist_traveled)
        print('centers: %s' % self.centers)
        print('total unlabeled: %s ' % len(unlabeled_data))
        print('total labels given: %s' % np.sum(self.num_labeled))
        return labels_for_unlabeled_data

    def closest_cluster_index(self, vec, threshold):
        index = -1
        min_dist = sys.maxsize
        for i, center in enumerate(self.centers):
            dist = np.linalg.norm(center - vec, 2)
            if dist < min_dist and dist <= threshold:
                min_dist = dist
                index = i
        return index

    def recompute_centers(self):
        for i, points in self.cluster_pts.items():
            new_center = np.sum(points[:, 1:], axis=0) / points.shape[0]-1
            old_center = self.centers[i]
            dist = np.linalg.norm(new_center - old_center, 2)
            self.centers[i] = new_center
            self.dist_traveled[i] += dist

    def c_score(self):
        return np.sum(np.divide(self.num_labeled, self.dist_traveled))

    def get_cluster_pts(self):
        return self.cluster_pts

    def get_full_pts(self):
        return np.vstack(list(self.cluster_pts.values()))
