import scipy.io
import numpy
import sklearn.decomposition
from sklearn import cluster
from sklearn.metrics.cluster import supervised
from scipy.optimize import linear_sum_assignment


alpha = 1.0

data = scipy.io.loadmat('digits-train.mat')
images_digits_raw = numpy.transpose(data['images_train'])   # Taking the transpose of data matrix

C = numpy.zeros([5000,5000])        # Creating the C array
# images_digits_row = numpy.zeros([5000])


for i in range(0, 5000):            # Looping through the data matrix to remove the ith column
    images_digits_row = images_digits_raw[i,:].reshape([1,-1])
    images_digits_final = numpy.delete(images_digits_raw, i, 0)
    Cj = sklearn.decomposition.sparse_encode(images_digits_row, images_digits_final, alpha=alpha)
    Cj = numpy.concatenate([Cj[:,:i], [[0.]], Cj[:,i:]], axis=1)
    C[i,:] = Cj             # Filling in the empty C array using column stack

W = numpy.abs(C.T) + numpy.abs(C)

## Spectral Clustering
spec = cluster.SpectralClustering(n_clusters=5, affinity='precomputed')
# spec.fit(W)
labels_true = data['labels_train']
labels_pred = spec.fit_predict(W)
labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
# labels_true : int array with ground truth labels, shape = [n_samples]
# labels_pred : int array with estimated labels, shape = [n_samples]
value = supervised.contingency_matrix(labels_true, labels_pred)
# value : array of shape [n, n] whose (i, j)-th entry is the number of samples in true class i and in predicted class j
[r, c] = linear_sum_assignment(-value)
accr = value[r, c].sum() / len(labels_true)
print(accr)
