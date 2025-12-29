import numpy as np

def std_PCA(W,d):
    """
    Compute the top principal directions from PCA

    This function performs a standard eigen-decomposition of a square matrix
    (e.g., a covariance matrix W = AᵀA) and returns the leading `d` eigenvectors
    corresponding to the largest eigenvalues.

    Parameters
    ----------
    W : np.ndarray
        Square matrix of shape (n_features, n_features). Typically this is a
        covariance or Gram matrix such as AᵀA derived from a data matrix A.
    d : int
        Number of principal directions (eigenvectors) to return.

    Returns
    -------
    eigenvectors : np.ndarray
        Matrix of shape (n_features, d) whose columns are the top `d` eigenvectors
        of `W`, ordered by descending eigenvalue.
    """
    [eigenValues,eigenVectors] = np.linalg.eig(W)

    #sort eigenvalues and eigenvectors in decending orders
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    #take the first d vectors. Obtained the solution
    return eigenVectors[:,:d]