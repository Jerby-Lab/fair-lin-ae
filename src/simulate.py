import numpy as np
from scipy.stats import nbinom, bernoulli

def mu_to_prob(size, mu):
    """
    Convert mean (mu) to probability of success (prob) for negative binomial distribution.

    Parameters
    ----------
    size : float
        The dispersion parameter.
    mu : float
        The mean of the distribution.

    Returns
    -------
    float
        The probability of success in each trial.
    """
    return size / (size + mu)

def sample_counts(N_samples, N_genes, size_gene, size_celltype):
    """
    Sample counts for one cell type using a negative binomial distribution.

    Parameters
    ----------
    N_samples : int
        The number of samples to generate.
    N_genes : int
        The number of genes.
    size_gene : float
        The size parameter for sampling gene means.
    size_celltype : float
        The size parameter for sampling cell counts.

    Returns
    -------
    np.ndarray
        A count matrix of shape (N_samples, N_genes).
    """

    # Sample gene means
    mu_gene = nbinom.rvs(size_gene, mu_to_prob(size_gene, 3), size=N_genes)
    
    # Initialize counts matrix
    counts = np.zeros((N_samples, N_genes), dtype=int)
    
    # Sample count for each cell using gene means
    for j in range(N_genes):
        prob = mu_to_prob(size_celltype, mu_gene[j])
        counts[:, j] = nbinom.rvs(size_celltype, prob, size=N_samples)
    
    return counts

def sample_counts_multiple_celltypes(N_samples_per_celltype: list, 
                                     N_genes: int, 
                                     size_gene: float, 
                                     size_celltype: float) -> np.ndarray:
    """
    Sample counts for multiple cell types by calling the sample_counts function multiple times.

    Parameters
    ----------
    N_samples_per_celltype : list of int
        A list of sample sizes for each cell type.
    N_genes : int
        The number of genes.
    size_gene : float
        The size parameter for sampling gene means.
    size_celltype : float
        The size parameter for sampling cell counts.

    Returns
    -------
    np.ndarray
        A count matrix with samples from all cell types combined.
    """
    list_counts = [sample_counts(N_samples, N_genes, size_gene, size_celltype) for N_samples in N_samples_per_celltype]
    counts = np.vstack(list_counts)
    return counts

def sample_counts_sig(N_samples, N_genes, size_gene, size_celltype):
    """
    Sample counts for one cell type and identify significant genes.

    Parameters
    ----------
    N_samples : int
        The number of samples to generate.
    N_genes : int
        The number of genes.
    size_gene : float
        The size parameter for sampling gene means.
    size_celltype : float
        The size parameter for sampling cell counts.

    Returns
    -------
    tuple
        A tuple containing the count matrix and the indices of significant genes.
    """
    mu_gene = nbinom.rvs(size_gene, mu_to_prob(size_gene, 3), size=N_genes)
    
    counts = np.zeros((N_samples, N_genes), dtype=int)
    for j in range(N_genes):
        counts[:, j] = nbinom.rvs(size_celltype, mu_to_prob(size_celltype, mu_gene[j]), size=N_samples)

    index_canonical = np.where(mu_gene > 0)[0]
        
    return counts, index_canonical
    
def sample_counts_multiple_celltypes_sig(N_samples_per_celltype, N_genes, size_genes_per_celltype, size_celltype):
    """
    Sample counts for multiple cell types and identify significant genes.

    Parameters
    ----------
    N_samples_per_celltype : list of int
        A list of sample sizes for each cell type.
    N_genes : int
        The number of genes.
    size_gene : float
        The size parameter for sampling gene means.
    size_celltype : float
        The size parameter for sampling cell counts.
    n_sig_genes_per_celltype : list of int
        A list specifying the number of signature genes for each cell type.

    Returns
    -------
    dict
        A dictionary containing the combined count matrix ('X') and a list of significant gene indices for each cell type ('canonical').
    """
    list_counts = []
    list_canonical = []
    for N_samples, size_gene in zip(N_samples_per_celltype, size_genes_per_celltype):
        counts, index_canonical = sample_counts_sig(N_samples, N_genes, size_gene, size_celltype)
        list_canonical.append(index_canonical)
        list_counts.append(counts)
    counts = np.vstack(list_counts)
    return {"X": counts, "sig": list_canonical}
    
def add_noise(mat: np.ndarray, mu: int, size: float) -> np.ndarray:
    """
    Add noise to a count matrix using a negative binomial distribution.

    Parameters
    ----------
    mat : np.ndarray
        The input count matrix.
    mu : int
        The mean count for the noise.
    size : float
        The size parameter for the negative binomial distribution.

    Returns
    -------
    np.ndarray
        The count matrix with added noise.
    """
    noise = nbinom.rvs(size, mu_to_prob(size, mu), size=mat.shape)
    return mat + noise

def create_dropout_mask(mat: np.ndarray, prob: float) -> np.ndarray:
    """
    Create a dropout mask for a count matrix.

    Parameters
    ----------
    mat : np.ndarray
        The input count matrix.
    prob : float
        The probability of dropout.

    Returns
    -------
    np.ndarray
        A mask matrix with the same dimensions as the input matrix.
    """
    N, M = mat.shape
    bool_drop = bernoulli.rvs(prob, size=N * M)
    mask_drop = np.reshape(bool_drop, (N, M))
    mask_drop = np.abs(mask_drop - 1)
    return mask_drop

def generate_celltype_labels(N_samples_per_celltype: list) -> np.ndarray:
    """
    Generate labels for different cell types.

    Parameters
    ----------
    N_samples_per_celltype : list of int
        A list of sample sizes for each cell type.

    Returns
    -------
    np.ndarray
        An array of labels corresponding to the cell types.
    """
    labels = np.concatenate([np.full(N_samples, i + 1) for i, N_samples in enumerate(N_samples_per_celltype)])
    return labels

