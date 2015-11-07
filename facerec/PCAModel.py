'''
PCAModel.py

Class for computing and storing principal components for dimensionality
reduction.

'''

import numpy as np


class PCAModel:

    def __init__(self, k_rank=0):
        '''
        Constructor for PCAModel.

        Args (optional):
            k_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''

        self.k_rank = k_rank
        self.is_fitted = False
        self.trn_mean = 0


    def center(self, faces):
        '''
        Centers data to zero-mean.

        Args:
            faces (numpy.ndarray): Data to center.
        Returns:
            numpy.ndarray, the normalized data.
        '''

        out = np.array(faces, copy=True, dtype='float')

        mean = np.mean(out, axis=1)
        for x in out.T:
            x -= mean
        return out, mean


    def fit(self, data):
        '''
        Performs SVD to find the top-k eigenvectors and learn an optimal
        subspace for the data.

        Args:
            data (numpy.ndarray): Data matrix of feature vectors.
        '''

        data, self.trn_mean = self.center(data)

        # Perform SVD to get eigenvectors
        U,S,V = np.linalg.svd(data, full_matrices=False)

        # Store top-k eigenvectors as components
        if self.k_rank != 0:
            self.components = U[0:self.k_rank]
        else:
            self.components = U

        self.is_fitted = True


    def transform(self, data):
        '''
        Transforms a feature vector by projecting it into the optimal subspace
        learned by PCA. Can be either a single vector or multiple.

        Args:
            data (numpy.ndarray): Data matrix of feature vectors.
        Returns:
            numpy.ndarray, the transformed matrix if the model has been fitted.
                None otherwise.
        '''

        if self.is_fitted:
            return self.components.T.dot(data-self.trn_mean)
        else:
            return None


