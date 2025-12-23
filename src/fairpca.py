import pandas as pd
import numpy as np

from standard_PCA import std_PCA
from utils import *

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


def mse_loss_per_row(X, X_hat):
    # Ensure X and X_hat have the same shape
    assert X.shape == X_hat.shape, "Shapes of X and X_hat must be the same"
    
    # Compute the MSE loss for each row
    mse_per_row = np.mean((X - X_hat) ** 2, axis=1)
    return mse_per_row

def get_loss(data, d, U, optimal_error):
    
    a = U @ U.T
    error = np.linalg.norm(data-np.dot(data, a), 'fro')**2 
    
    return (error-optimal_error)/len(data)


def my_fair_pca(data, d, init_u, t):
    """
    data: the full data
    d: the target dimension
    """
    
    np.random.seed(1)
    full_dimension = len(data[0][0])
    U = init_u
    num_groups = len(data)
    lr = 0.001
    
    opt_err = []
    cov = []
    for i in range(num_groups): 
        cov_i = data[i].T @ data[i]
        cov.append(cov_i)
        P_all_i = std_PCA(cov_i/len(cov_i), len(data[i][0]))
        p_i = P_all_i[:,:d] @ P_all_i[:,:d].T
        opt_err.append(np.linalg.norm(data[i]-np.dot(data[i], p_i), 'fro')**2)
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
	data1 =  np.load(path)
	X = data1["X"]
	labels = data1["labels"]
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# do normal pca
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

	# initialize U 
	Cov = X_scaled.T @ X_scaled / len(X_scaled) # compute covariance matrix 
	n=len(Cov[0]) # number of dimensions of the data 
	P_all = std_PCA(Cov,n) 
	U = P_all[:,:d]

	gradient_project_m = []
	d=30
	for t in [5, 10, 200, 1, 100]:
		gradient_project_m.append(my_fair_pca(data, d, U, t))
		np.savez("../results/{}_U_t{}.npz".format(name, str(t)), U=gradient_project_m)

run("data/sim3_ct_cc.npz", "sim3")
run("data/sim1.npz", "sim1")


