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
    umap_model = umap.UMAP(n_components=2, random_state=seed)
    umap_embedding = umap_model.fit_transform(latent)
    df_umap = pd.DataFrame({'umap1': umap_embedding[:, 0], 'umap2': umap_embedding[:, 1], 'labels': labels.astype(str)})
    sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='labels', palette='tab10', ax=ax, s = 8, edgecolor=None)
    ax.set_title('UMAP projection of 30 PCs ({})'.format(ltype))
    return(df_umap)

def plot_diff_labels(ax, df_umap, labels, ltype="Original Labels"):
    df_umap['labels'] = labels.astype(str)
    sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='labels', palette='tab10', ax=ax, s = 8, edgecolor=None)
    ax.set_title('UMAP projection of 30 PCs ({})'.format(ltype))

def plot_recon(ax, X_scaled, X_hat, labels):
    mse = mse_loss_per_row(X_scaled, X_hat)
    data = pd.DataFrame({'Loss': mse, 'labels': labels})
    sns.boxplot(x='labels', y='Loss', data=data, palette="tab10", ax=ax)
    ax.set_title('Reconstruction Loss by Cell Group')
    ax.set_xlabel('Cell group')
    ax.set_ylabel('Reconstruction Loss')

def loss_plot(ax, losses):
    ax.plot(losses)
    # ax.set_xticks(np.arange(range(10, 50, 10)), range(10, 50, 10))
    ax.set_ylabel('Reconstruction loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Loss over Epochs')
    
    
def plot_cm(ax, labels, clusters):
    df = pd.DataFrame({'list1': labels, 'list2': clusters + 1})
    out = pd.DataFrame(np.array(pd.crosstab(df['list1'], df['list2'])).T/[5000,4000,3500,3000,50], 
                      index = list(set(clusters+1)), 
                      columns = list(set(labels)))
    sns.heatmap(out*100, annot=True, fmt=".4f", cmap='viridis', ax=ax)
    ax.set_ylabel('Post-Cluster Labels')
    ax.set_xlabel('Original Labels')
    ax.set_title('Confusion Matrix')

def cluster_nn(X_reduc, ctype = "louvain", resolution=0.8):
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
    kmeans = KMeans(n_clusters=n_clus, random_state=0)
    kmeans.fit(X_reduc)
    
    # Extract cluster labels
    kmeans_labels = kmeans.labels_
    return kmeans_labels 

def test_hypers(X_scaled, labels, hypers, analytical_pc, dro_loss=KL_loss, dro_type="KL"):
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
    # Ingest test
    data1 =  np.load('data/synthetic1.npz')
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
