import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import seaborn as sns
import umap
import scanpy as sc

from standard_PCA import std_PCA
from utils import *

def mse_loss_per_row(X, X_hat):
	"""
	Compute per-sample mean squared reconstruction error.

	This function calculates the mean squared error (MSE) between an original
	data matrix and its reconstruction, independently for each row/sample.

	Parameters
	----------
	X : np.ndarray
		Original data matrix of shape (n_samples, n_features).
	X_hat : np.ndarray
		Reconstructed data matrix of shape (n_samples, n_features).
		Must have the same shape as `X`.

	Returns
	-------
	mse_per_row : np.ndarray
		One-dimensional array of shape (n_samples,) containing the mean squared
		reconstruction error for each sample.
	"""

	# ensure X and X_hat have the same shape
	assert X.shape == X_hat.shape, "Shapes of X and X_hat must be the same"
	
	# compute the MSE loss for each row
	mse_per_row = np.mean((X - X_hat) ** 2, axis=1)
	return mse_per_row

def get_loss(data, d, U, optimal_error):
	"""
	Compute normalized excess reconstruction loss for a single group.

	This function measures how much worse a given projection performs on a
	group of samples relative to a precomputed optimal (baseline) reconstruction
	error, normalized by the number of samples in the group.

	Parameters
	----------
	data : np.ndarray
	    Data matrix for a single group, of shape (n_samples_group, n_features).
	d : int
	    Target subspace dimension. Included for interface consistency; not
	    explicitly used in the computation.
	U : np.ndarray
	    Projection matrix of shape (n_features, d), whose columns define the
	    learned subspace.
	optimal_error : float
	    Baseline squared Frobenius reconstruction error for this group (e.g.,
	    from standard PCA fitted on the group alone).

	Returns
	-------
	loss : float
	    Normalized excess reconstruction error, defined as
	    (||data - data U Uᵀ||_F² − optimal_error) / n_samples_group.
	"""

	a = U @ U.T
	error = np.linalg.norm(data-np.dot(data, a), 'fro')**2 
	
	return (error-optimal_error)/len(data)


def my_fair_pca(data, d, init_u, t):
	"""
	Learn a fairness-aware PCA projection via gradient-based optimization.

	This function implements a FairPCA-style algorithm that minimizes a
	softmax-weighted combination of group-specific reconstruction errors.
	Groups with higher excess loss receive larger weights, controlled by a
	temperature parameter.

	Parameters
	----------
	data : list of np.ndarray
	    List of group-specific data matrices. Each element has shape
	    (n_samples_group, n_features).
	d : int
	    Number of principal components (subspace dimension) to learn.
	init_u : np.ndarray
	    Initial projection matrix of shape (n_features, d), typically obtained
	    from standard PCA on pooled data.
	t : float
	    Temperature parameter controlling emphasis on high-loss groups.
	    Larger values focus optimization more strongly on the worst-off groups.

	Returns
	-------
	U : np.ndarray
	    Learned projection matrix of shape (n_features, d)
	"""

	# initialize parameters
	np.random.seed(1)
	full_dimension = len(data[0][0])
	U = init_u
	num_groups = len(data)
	lr = 0.001
	
	# initialize global optimal fit per group
	opt_err = []
	cov = []
	for i in range(num_groups): 
		cov_i = data[i].T @ data[i]
		cov.append(cov_i)
		P_all_i = std_PCA(cov_i/len(cov_i), len(data[i][0]))
		p_i = P_all_i[:,:d] @ P_all_i[:,:d].T
		opt_err.append(np.linalg.norm(data[i]-np.dot(data[i], p_i), 'fro')**2)
	
	# gradient based fit 
	losses = []
	for i in range(40000):
		q, r = np.linalg.qr(U)
		inverse_r =  np.linalg.inv(r)
		
		loss_each_i = [get_loss(data[i], d, U, opt_err[i]) for i in range(num_groups)] 
		max_l = max(loss_each_i) # calculate the maximum loss between these two
		grads = [-1*np.dot(cov_i, U) for cov_i in cov] # calculate the gradients based on covariance matrix and np
		weights = [np.exp(t*(loss_i - max_l)) for loss_i in loss_each_i] 
		if i % 1000 == 0:
			df = pd.DataFrame([loss_each_i, weights/np.sum(weights)], index = ["Disparity Error", "Weights"], columns= range(1, 6))
			print(df)
		grad = np.sum(np.array(weights)[:, np.newaxis, np.newaxis] * grads, axis=0)
		grad = grad/np.sum(weights*np.array([len(dat) for dat in data]))
		U = np.dot(U - lr * grad, inverse_r)
	return U

def run(path, name):
	"""
	Run standard PCA and FairPCA experiments and save results to disk.

	This function loads data from a `.npz` file, performs standard PCA to compute
	baseline reconstructions and errors, then runs FairPCA with multiple
	temperature settings and saves the learned projection matrices.

	Parameters
	----------
	path : str
	    Path to a `.npz` file containing:
	    - `X`: np.ndarray of shape (n_samples, n_features)
	    - `labels`: np.ndarray of shape (n_samples,) indicating group membership.
	name : str
	    Identifier used to construct output filenames.
	"""

	# ingest data
	data1 =  np.load(path)
	X = data1["X"]
	labels = data1["labels"]
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# standard PCA
	d = 30
	pca = PCA(n_components=d)
	pc = pca.fit_transform(X_scaled)
	X_hat = pca.inverse_transform(pc)
	mse = mse_loss_per_row(X_scaled, X_hat)
	np.savez("../results/{}_pca.npz".format(name), X_hat=X_hat)

	# prep data
	unique_labels = np.unique(labels)
	data = [X_scaled[labels == label] for label in unique_labels]
	mses = [np.mean(mse[labels == label]) for label in unique_labels]
	mses.append(np.mean(mse))
	mses = [mses]

	# initialize U, the projection matrix
	Cov = X_scaled.T @ X_scaled / len(X_scaled) # compute covariance matrix 
	n=len(Cov[0]) # number of dimensions of the data 
	P_all = std_PCA(Cov,n) 
	U = P_all[:,:d]

	# run FairPCA with different tilt parameter t
	gradient_project_m = []
	d=30
	for t in [1, 5, 10,100,200]:
		gradient_project_m.append(my_fair_pca(data, d, U, t))
		np.savez("../results/{}_U_t{}.npz".format(name, str(t)), U=gradient_project_m)

run("../data/sim3_ct_cc.npz", "sim3")


