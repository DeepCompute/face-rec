'''
PCAModel.py

Class for computing and storing principal components for dimensionality
reduction.

'''

import numpy as np


class PCAModel:

    def __init__(self, dimensions=0, use_kernel=False, variance=10000):
        '''
        Constructor for PCAModel.

        Args (optional):
            dimensions (int): How many principal components to keep. A value
                of 0 indicates that it should keep all components.
            use_kernel (bool): Whether or not the model should use a kernel.
            variance (float): The variance of the RBF kernel, if used.
        '''

        self.dimensions = dimensions
        self.use_kernel = use_kernel
        self.variance   = variance

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


    def center_kernel(self, new_kernel):
        '''
        Centers the kernel matrix with respect to the training one.

        Args:
            new_kernel (numpy.ndarray): The kernel to center.
        Returns:
            numpy.ndarray, the normalized kernel matrix.
        '''

        d, m = new_kernel.shape
        d, n = self.kernel.shape

        one_m = np.empty((n,m))
        one_n = np.empty((n,n))

        one_m.fill(1.0/n)
        one_n.fill(1.0/n)

        return new_kernel - self.kernel.dot(one_m) - one_n.dot(new_kernel) \
                + one_n.dot(self.kernel).dot(one_m)


    def kernel_matrix(self, data1, data2):
        '''
        Create a kernel matrix using a radial basis function kernel.

        Args:
            data (numpy.ndarray): The data to create a kernel from.
        Return:
            numpy.ndarray: The kernel matrix.
        '''

        N1 = data1.shape[1] if len(data1.shape) == 2 else 1
        N2 = data2.shape[1] if len(data2.shape) == 2 else 1

        gram = np.empty((N1, N2))

        for y in range(0, N2):
            for x in range(0, N1):

                d1 = data1[:,x] if N1 != 1 else data1
                d2 = data2[:,y] if N2 != 1 else data2

                top = np.linalg.norm(d1-d2) ** 2

                gram[x,y] = np.exp( -top/(2*self.variance**2) )

        return gram


    def fit(self, data):
        '''
        Performs SVD to find the top-k eigenvectors and learn an optimal
        subspace for the data.

        Args:
            data (numpy.ndarray): Data matrix of feature vectors.
        '''

        self.data = np.array(data, dtype='float64')

        if self.use_kernel:
            self.kernel = self.kernel_matrix(self.data, self.data)
            data = self.center_kernel(self.kernel)
        else:
            data, self.trn_mean = self.center(self.data)

        # Perform SVD to get eigenvectors
        U,S,V = np.linalg.svd(data, full_matrices=False)


        # Store eigenvectors as components
        self.components = U


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

        if self.components is None:
            raise RuntimeError('PCAModel has not been fitted yet!')

        if self.dimensions != 0:
            reduced_components = self.components[:,0:self.dimensions]
        else:
            reduced_components = self.components

        if self.use_kernel:
            data = self.kernel_matrix(self.data, data)
            data = self.center_kernel(data)
            return reduced_components.T.dot(data)
        else:
            return reduced_components.T.dot(data-self.trn_mean)


