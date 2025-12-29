import torch
import torch.nn as nn
import torch.optim as optim

import scanpy as sc

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import umap
import time
import logging
import sys

from models import LinearAE
import simulate
from utils import *
from losses import CVaR_loss, KL_loss

def train_autoencoder(model, data, criterion, epochs=3, learning_rate=1e-4, regularization=1e-3):
    """
    Train an autoencoder model using a provided specific DRO loss.

    This routine runs a standard PyTorch training loop over a DataLoader and
    optimizes the parameters of an encoder/decoder model using Adam with L2
    weight decay applied to both submodules.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder model that maps inputs of shape (batch_size, n_features)
        to reconstructions of the same shape via `model(inputs)`.
    data : torch.utils.data.DataLoader
        Iterable over minibatches. Each batch is expected to be either a tensor
        of shape (batch_size, n_features) or something that can be reshaped to
        (batch_size, -1).
    criterion : callable
        Loss function taking `(outputs, inputs)` and returning a scalar tensor.
        This can be standard MSE or a DRO-style loss (e.g., KL-loss / CVaR-loss)
        that internally computes a robustified reconstruction objective.
    epochs : int, default=3
        Number of full passes over the dataset.
    learning_rate : float, default=1e-4
        Learning rate for the Adam optimizer.
    regularization : float, default=1e-3
        Weight decay (L2 regularization) applied to both encoder and decoder
        parameters via Adam's `weight_decay`.

    Returns
    -------
    model : torch.nn.Module
        The trained model (same object as input, returned for convenience).
    losses : list of float
        List of average epoch losses, one value per epoch.

    Side Effects
    ------------
    Prints the epoch-level average loss and displays a tqdm progress bar for
    each epoch.

    Notes
    -----
    - Inputs are flattened via `inputs.view(inputs.size(0), -1)` before passing
      to the model.
    - The epoch loss is computed as the mean of `loss.item()` over minibatches.
    """

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'weight_decay': regularization},
        {'params': model.decoder.parameters(), 'weight_decay': regularization}
    ], lr=learning_rate)

    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(data), total=len(data), mininterval=5.0, miniters=100)
        for i, batch in progress_bar:
            inputs = batch
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(data)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
    return model, losses

def plot_pcs(ax, ls, X_scaled, labels, title, pc):
    """
    Scatter-plot the first two principal components with label-based coloring.

    This helper aligns the signs of the provided PC coordinates to a reference
    PC solution (via `align_sign`) to make visual comparisons across runs more
    consistent, then plots PC1 vs PC2 using seaborn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    ls : np.ndarray
        Reference low-dimensional coordinates used to align signs, of shape
        (n_samples, >=2). Typically a reference PC embedding.
    X_scaled : np.ndarray
        Scaled input data matrix of shape (n_samples, n_features). Used for
        annotation only (gene count in title).
    labels : np.ndarray
        Array of shape (n_samples,) containing integer/str group labels for
        coloring and legend counts.
    title : str
        Unused label for the plot title (kept for API compatibility).
    pc : np.ndarray
        PC coordinates to plot, of shape (n_samples, >=2).

    Returns
    -------
    None
    """

    df_plot = pd.DataFrame({'pc1': align_sign(pc[:,0], ls[:, 0]), 'pc2': align_sign(pc[:,1], ls[:, 1]), 'labels': labels.astype(str)})
    sns.scatterplot(data=df_plot, x='pc1', y='pc2', hue='labels', palette='tab10', ax=ax, s = 8, edgecolor=None)
    ax.set_title(f"Cells in PC space ({X_scaled.shape[1]} genes)")
    
    # Create a color palette
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    label_colors = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    # Create a legend
    handles = [plt.Line2D([0,0],[0,0], color=label_colors[label], marker='o', linestyle='') for label in unique_labels]
    labels_legend = [f'Label {label} (n={np.sum(labels == label)})' for label in unique_labels]
    ax.legend(handles, labels_legend, title='Cell Types', bbox_to_anchor=(1, 1), loc='upper right')

def plot_umap(ax, latent, labels, seed=888, ltype="Original Labels"):
    """
    Compute and plot a 2D UMAP embedding of a latent representation.

    This function fits a UMAP model to `latent`, projects to 2 dimensions, and
    renders a scatter plot colored by the provided labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    latent : np.ndarray
        Latent representation of shape (n_samples, n_latent) (e.g., PCs or
        learned embedding).
    labels : np.ndarray
        Array of shape (n_samples,) containing labels for coloring.
    seed : int, default=888
        Random seed passed to UMAP for reproducibility.
    ltype : str, default="Original Labels"
        Descriptor appended to the plot title (e.g., "Original Labels",
        "Louvain clusters", etc.).

    Returns
    -------
    df_umap : pandas.DataFrame
        DataFrame with columns ['umap1', 'umap2', 'labels'] containing the 2D
        embedding and label strings.
    """
    umap_model = umap.UMAP(n_components=2, random_state=seed)
    umap_embedding = umap_model.fit_transform(latent)
    df_umap = pd.DataFrame({'umap1': umap_embedding[:, 0], 'umap2': umap_embedding[:, 1], 'labels': labels.astype(str)})
    sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='labels', palette='tab10', ax=ax, s = 8, edgecolor=None)
    ax.set_title('UMAP projection of 30 PCs ({})'.format(ltype))
    return(df_umap)

def plot_diff_labels(ax, df_umap, labels, ltype="Original Labels"):
    """
    Re-color an existing UMAP embedding with an alternative label assignment.

    This helper is intended for comparing different clusterings on the same
    UMAP coordinates. It overwrites the 'labels' column of `df_umap` and redraws
    a scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    df_umap : pandas.DataFrame
        DataFrame containing at least columns ['umap1', 'umap2'] and typically
        an existing 'labels' column.
    labels : np.ndarray
        New labels of shape (n_samples,) to visualize on the same embedding.
    ltype : str, default="Original Labels"
        Descriptor appended to the plot title.

    Returns
    -------
    None
    """
    df_umap['labels'] = labels.astype(str)
    sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='labels', palette='tab10', ax=ax, s = 8, edgecolor=None)
    ax.set_title('UMAP projection of 30 PCs ({})'.format(ltype))

def plot_recon(ax, X_scaled, X_hat, labels):
    """
    Plot reconstruction error distributions across labeled groups.

    This function computes per-sample MSE reconstruction error using
    `mse_loss_per_row` and visualizes the distribution as a boxplot grouped by
    the provided labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    X_scaled : np.ndarray
        Original (typically scaled) data matrix of shape (n_samples, n_features).
    X_hat : np.ndarray
        Reconstructed data matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Group labels of shape (n_samples,) used for stratifying the boxplot.

    Returns
    -------
    None
    """
    mse = mse_loss_per_row(X_scaled, X_hat)
    data = pd.DataFrame({'Loss': mse, 'labels': labels})
    sns.boxplot(x='labels', y='Loss', data=data, palette="tab10", ax=ax)
    ax.set_title('Reconstruction Loss by Cell Group')
    ax.set_xlabel('Cell group')
    ax.set_ylabel('Reconstruction Loss')

def loss_plot(ax, losses):
    """
    Plot training loss trajectory over epochs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    losses : sequence of float
        Epoch-level loss values, typically returned from `train_autoencoder`.

    Returns
    -------
    None

    """
    ax.plot(losses)
    ax.set_ylabel('Reconstruction loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Loss over Epochs')
    
    
def plot_cm(ax, labels, clusters):
    """
    Plot a normalized confusion-style heatmap between original labels and clusters.

    This function builds a contingency table (original labels vs cluster labels),
    applies a custom column-wise normalization using a hard-coded denominator
    vector, and plots the result as a percentage heatmap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    labels : array-like
        Original labels of shape (n_samples,). Interpreted as the "true" groups.
    clusters : array-like
        Cluster assignments of shape (n_samples,). These are shifted by +1
        internally for display.

    Returns
    -------
    None
    """
    df = pd.DataFrame({'list1': labels, 'list2': clusters + 1})
    out = pd.DataFrame(np.array(pd.crosstab(df['list1'], df['list2'])).T/[5000,4000,3500,3000,50], 
                      index = list(set(clusters+1)), 
                      columns = list(set(labels)))
    sns.heatmap(out*100, annot=True, fmt=".4f", cmap='viridis', ax=ax)
    ax.set_ylabel('Post-Cluster Labels')
    ax.set_xlabel('Original Labels')
    ax.set_title('Confusion Matrix')

def cluster_nn(X_reduc, ctype = "louvain", resolution=0.8):
    """
    Cluster samples using a kNN graph and community detection (Louvain/Leiden).

    This function wraps Scanpy's neighborhood graph construction and applies
    either Louvain or Leiden clustering on the resulting graph.

    Parameters
    ----------
    X_reduc : np.ndarray
        Low-dimensional representation of shape (n_samples, n_features_reduc),
        such as PCs or latent factors.
    ctype : {"louvain", "leiden"}, default="louvain"
        Clustering algorithm to run. Any value other than "louvain" triggers
        Leiden clustering.
    resolution : float, default=0.8
        Resolution parameter controlling the granularity of the clustering.
        Larger values typically yield more clusters.

    Returns
    -------
    clusters : pandas.Series
        Cluster labels of length n_samples, cast to integer dtype.

    Notes
    -----
    - Uses `sc.pp.neighbors` with `n_neighbors=15` and default distance metric.
    - Returns cluster labels from `adata.obs['louvain']` or `adata.obs['leiden']`.
    """
    # Convert PCA results to AnnData object for Scanpy
    adata = sc.AnnData(X_reduc)
    # Compute the neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=15)

    if (ctype == "louvain"):
        # Perform Louvain clustering
        sc.tl.louvain(adata, resolution=resolution)   
        clusters = adata.obs['louvain'].astype(int)
    else:
        # Perform Leiden clustering
        sc.tl.leiden(adata, resolution=resolution)
        # Extract cluster labels
        clusters = adata.obs['leiden'].astype(int)
    return(clusters)

def cluster_kmeans(X_reduc, ctype = "louvain", n_clus=5):
    """
    Cluster samples using k-means in a low-dimensional space.

    Parameters
    ----------
    X_reduc : np.ndarray
        Low-dimensional representation of shape (n_samples, n_features_reduc).
    ctype : str, default="louvain"
        Unused argument retained for interface compatibility.
    n_clus : int, default=5
        Number of k-means clusters.

    Returns
    -------
    kmeans_labels : np.ndarray
        Integer cluster labels of shape (n_samples,).

    Notes
    -----
    Uses scikit-learn's KMeans with `random_state=0`.
    """
    kmeans = KMeans(n_clusters=n_clus, random_state=0)
    kmeans.fit(X_reduc)
    
    # Extract cluster labels
    kmeans_labels = kmeans.labels_
    return kmeans_labels 

def test_hypers(X_scaled, labels, hypers, analytical_pc, dro_loss=KL_loss, dro_type="KL"):
    """
    Sweep DRO hyperparameters for a linear autoencoder and generate diagnostics.

    This function trains a linear autoencoder multiple times, once per value in
    `hypers`, using either a KL-DRO or CVaR-DRO reconstruction loss. For each run,
    it saves the trained weights, produces a multi-panel diagnostic figure, and
    tracks summary statistics of per-sample reconstruction error.

    Parameters
    ----------
    X_scaled : np.ndarray
        Input data matrix of shape (n_samples, n_features), typically standardized.
    labels : np.ndarray
        Ground-truth group labels of shape (n_samples,) used for plotting and
        confusion matrix comparisons.
    hypers : sequence of float
        List of hyperparameter values to sweep. Interpreted as `beta` for CVaR
        or `Lambda` for KL, depending on `dro_type`.
    analytical_pc : np.ndarray
        Reference PCA coordinates (shape (n_samples, >=2)) used to align signs in
        PC plots (via `plot_pcs`).
    dro_loss : callable, default=KL_loss
        Loss class/factory used to create a criterion. Must be compatible with:
        - CVaR: dro_loss(beta=hyper, eta=0.1, device=device)
        - KL:   dro_loss(Lambda=hyper, gamma=0.95, device=device)
    dro_type : {"KL", "CVaR"}, default="KL"
        Determines how `dro_loss` is instantiated and how outputs are named.

    Returns
    -------
    mean_loss : list of float
        Mean per-sample reconstruction MSE for each hyperparameter setting.
    max_loss : list of float
        Maximum per-sample reconstruction MSE for each hyperparameter setting.
    min_loss : list of float
        Minimum per-sample reconstruction MSE for each hyperparameter setting.
    """
    dimension = 30
    input_dim = X_scaled.shape[1]
    loader = torch.utils.data.DataLoader(CustomDataset(X_scaled), batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dro_type)
    mean_loss =[]
    max_loss = []
    min_loss = []
    for hyper in hypers:
        print("hyper: {}".format(str(hyper)))
        #################
        # set up params #
        #################
        loader = torch.utils.data.DataLoader(CustomDataset(X_scaled), batch_size=4, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder = LinearAE(input_dim, dimension)
        if dro_type == "CVaR":
            criterion = dro_loss(beta=hyper, eta=0.1, device=device)
        else: 
            criterion = dro_loss(Lambda=hyper, gamma=0.95, device=device)

        ###############
        # train model #
        ###############
        trained_autoencoder, losses = train_autoencoder(autoencoder, loader, criterion, epochs=200)
        model_save_path = '../results/model_{}_{}.pth'.format(str(dro_type), str(hyper))
        torch.save(trained_autoencoder.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        ########
        # plot #
        ########
        w1, w2 = trained_autoencoder.encoder.weight.data.numpy(), trained_autoencoder.decoder.weight.data.numpy()

        p_linear_ae, _, _ = np.linalg.svd(w2, full_matrices=False)
        latent_space= np.dot(X_scaled, p_linear_ae)
        X_hat = X_scaled@p_linear_ae@p_linear_ae.T
        
        fig, axes = plt.subplots(2, 4, figsize=(38, 20))
        
        # Call the plot functions with the corresponding axes
        plot_pcs(axes[0, 0], latent_space, X_scaled, labels, 'PC Plot', analytical_pc)  # pc plot
        df_umap = plot_umap(axes[0, 1], latent_space, labels)  # umap
        plot_recon(axes[1, 1], X_scaled, X_hat, labels)  # reconstruction per group
        loss_plot(axes[1, 0], losses)  # Another umap plot example
        
        clusters1 = cluster_nn(latent_space, resolution=1.2)
        plot_diff_labels(axes[0,2], df_umap, labels=np.array(clusters1.to_list())+1, ltype="Louvain Clusters (res=1.2)")
        plot_cm(axes[1,2], labels, clusters1) 
        
        clusters2 = cluster_kmeans(latent_space, n_clus=5)
        plot_diff_labels(axes[0,3], df_umap, labels=clusters2+1, ltype="k-Means Clusters (n=5)")
        plot_cm(axes[1,3], labels, clusters2)
        
        plt.tight_layout()
        
        recon = mse_loss_per_row(X_scaled, X_hat)
        fig.savefig('../results/linearae_{}_{}.pdf'.format(str(dro_type), str(hyper)))

        ########
        # save #
        ########
        mean_loss.append(np.mean(recon))
        max_loss.append(np.max(recon))
        min_loss.append(np.min(recon))

        np.savez("../results/{}_losses_{}.npz".format(str(dro_type), str(hyper)), 
            losses = [mean_loss, max_loss, min_loss])

    return mean_loss, max_loss, min_loss

def main(): 
    """
    Main function to run hyperparameter testing of Linear Autoencoders with
    the DRO losses: CVaR and KL-DRO
    """
    # Ingest dataset
    data1 =  np.load('../data/sim3_ct_cc.npz')
    X = data1["X"]
    labels = data1["labels"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce dimensions
    pca = PCA(n_components=30)
    pc = pca.fit_transform(X_scaled)

    # Test hyperparameters 
    mean_loss, max_loss, min_loss = test_hypers(X_scaled, labels, 
                hypers=[0.005, 0.01, 0.0001, 0.0005, 0.001, 0.05, 0.1, 0.15, 0.2, 0.3],
                analytical_pc = pc, dro_loss=CVaR_loss, dro_type = "CVaR")
    np.savez("../results/cvar_losses.npz", hypers =[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
             mean_l=mean_loss, max_l=max_loss, min_l=min_loss)
    mean_loss, max_loss, min_loss = test_hypers(X_scaled, labels, 
                hypers=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
                analytical_pc = pc, dro_loss=KL_loss, dro_type = "KL")
    np.savez("../results/kl_losses.npz", hypers =[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
             mean_l=mean_loss, max_l=max_loss, min_l=min_loss)


main()
