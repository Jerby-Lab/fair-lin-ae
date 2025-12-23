class tab_dataset(torch.utils.data.Dataset):
    """
    Simple Torch `Dataset` wrapper for tabular data stored as a NumPy array.

    This dataset converts the provided NumPy array to a `torch.FloatTensor`
    once at initialization and returns rows by index.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features). Will be cast to `np.float32`
        and converted to a torch tensor.

    Attributes
    ----------
    X : torch.FloatTensor
        Tensor of shape (n_samples, n_features).

    Notes
    -----
    `__len__` uses `len(self.X)` when available; otherwise it falls back to
    `self.X.shape[0]`. For torch tensors, `len(self.X)` is usually defined.
    """

    def __init__(self, X):
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        """Return the number of rows (samples) in the dataset."""
        try:
            L = len(self.X)
        except Exception:
            L = self.X.shape[0]
        return L

    def __getitem__(self, idx):
        """
        Return a single sample (row) by index.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        torch.FloatTensor
            Tensor of shape (n_features,) corresponding to row `idx`.
        """
        return self.X[idx]


class CustomDataset(Dataset):
    """
    Torch `Dataset` that lazily converts individual samples to float tensors.

    Parameters
    ----------
    data : Sequence or np.ndarray
        Container of samples where `data[idx]` returns an array-like object.

    Notes
    -----
    Unlike `tab_dataset`, conversion to `torch.tensor(..., dtype=torch.float32)`
    happens in `__getitem__` for each sample.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        """
        Return the number of samples.

        Returns
        -------
        int
            Number of items in `self.data`.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sample as a float tensor.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        torch.FloatTensor
            Tensor representation of `self.data[idx]`.
        """
        return torch.tensor(self.data[idx], dtype=torch.float32)


def geo_mean_through_log(numberList):
    """
    Compute the geometric mean using logs for numerical stability.

    Uses log-space aggregation to reduce overflow/underflow compared to
    multiplying numbers directly.

    Parameters
    ----------
    numberList : array-like
        List/array of nonnegative numbers.

    Returns
    -------
    float
        Geometric mean of `numberList`. If any entry is <= 1e-12, returns 0.

    Notes
    -----
    The early-return threshold (1e-12) is a practical guard against `log(0)`
    and extremely small values.
    """
    #if some is 0, return 0.
    if (np.amin(numberList) <= 1.e-12):
        return 0

    logNumberList = np.log(numberList)
    return np.exp(logNumberList.sum()/len(numberList))


def scaleVar(dataframe, colArray):
    """
    Scale a set of columns by a *shared* standard deviation.

    This normalizes multiple columns "together" by dividing all selected
    columns by the standard deviation computed over the pooled values across
    those columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe (modified in-place).
    colArray : array-like of str
        Column names to scale jointly.

    Returns
    -------
    None
        Operates in-place. If pooled standard deviation is 0, does nothing.

    Notes
    -----
    This is useful when the columns share a common unit/meaning (e.g., the same
    quantity measured across multiple months) and should be scaled consistently.
    """
    SD = dataframe[colArray].stack().std()  # pooled SD
    if SD == 0:
        return
    dataframe[colArray] = dataframe[colArray] / SD


def scaleVarOneCol(dataframe, nameStr):
    """
    Scale a single column to have unit standard deviation (mean unchanged).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe (modified in-place).
    nameStr : str
        Name of the column to scale.

    Returns
    -------
    None
        Operates in-place. If the column standard deviation is 0, does nothing.
    """
    sd = dataframe[nameStr].std()
    if sd == 0:
        return
    dataframe[nameStr] = dataframe[nameStr] / sd


def input_check(n, k, d, B, function_name='the function'):
    """
    Validate dimensions and basic consistency for fair PCA utilities.

    Checks that:
    - `k >= 1`
    - `B` contains at least `k` matrices
    - each `B[i]` has shape (n, n) for i in [0, k-1]
    - `d` is an integer in [1, n]

    Parameters
    ----------
    n : int
        Number of original features (dimension of each PSD matrix).
    k : int
        Number of groups.
    d : int
        Target subspace dimension (rank).
    B : list of np.ndarray
        List of group-specific PSD matrices, each of shape (n, n).
    function_name : str, default='the function'
        Name used in error messages for easier debugging.

    Returns
    -------
    int
        0 if all checks pass; otherwise a positive error code.

    Notes
    -----
    This function prints human-readable error messages and returns a code
    rather than raising exceptions.
    """
    if (isinstance(function_name, str) == False):
        print("Error: check_input is used with function name that is not string. Exit the check.")
        return 1

    if (k < 1):
        print("Error: " + function_name + " is called with k<1.")
        return 2

    if (len(B) < k):
        print("Error: " + function_name + " is called with not enough matrices in B.")
        return 3

    for i in range(k):
        if (B[i].shape != (n, n)):
            print("Error: " + function_name + " is called with input matrix B_i not the correct size."
                  + "Note: i=" + str(i) + " , starting indexing from 0 to k-1")
            return 4

    if (((d > 0) and (d <= n)) == False):
        print("Error: " + function_name + " is called with invalid value of d, which should be a number between 1 and n inclusive.")
        return 5

    return 0


def getObj(n, k, d, B, X):
    """
    Compute group-wise variance / loss objectives for a given projection matrix.

    Given group PSD matrices `B_1, ..., B_k` (typically covariance-like matrices)
    and a projection matrix `X` (commonly an n-by-n PSD matrix representing a rank-d
    projector), this computes:

    - Group variance:      Var_i = <B_i, X>
    - Group best variance: Best_i = sum of top-d eigenvalues of B_i
    - Group loss:          Loss_i = Var_i - Best_i
    - Aggregate objectives:
        * MM_Var    = min_i Var_i
        * MM_Loss   = min_i Loss_i   (note: often "min max loss" is used elsewhere;
                      here you compute min over i of (Var_i - Best_i))
        * NSW       = geometric mean of Var_i
        * Total_Var = sum_i Var_i

    Parameters
    ----------
    n : int
        Feature dimension (B_i are n-by-n).
    k : int
        Number of groups.
    d : int
        Target subspace rank/dimension.
    B : list[np.ndarray]
        List of k (or more) PSD matrices of shape (n, n). Only the first k are used.
        Matrices should be centered appropriately for your variance interpretation.
    X : np.ndarray
        Candidate solution matrix of shape (n, n). Typically a rank-d projector, but
        this function will still compute objectives even if X is not PSD/symmetric.

    Returns
    -------
    dict
        Dictionary containing:
        - 'MM_Var', 'MM_Loss', 'NSW', 'Total_Var'
        - per-group metrics: 'Loss{i}', 'Var{i}', 'Best{i}' for i=0..k-1

        Returns -1 if the input check fails.

    Warnings
    --------
    Prints a warning if `rank(X) != d`.

    Notes
    -----
    Inner products are computed as elementwise sums: <A, X> = sum_{i,j} A_ij X_ij.
    """
    if (input_check(n, k, d, B, function_name='fairDimReductionFractional') > 0):
        return -1

    if (np.linalg.matrix_rank(X) != d):
        print("Warning: getObj is called with X having rank not equal to d.")

    obj = dict()

    best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
    loss = [np.sum(np.multiply(B[i], X)) - best[i] for i in range(k)]
    var  = [np.sum(np.multiply(B[i], X)) for i in range(k)]

    obj.update({
        'MM_Var': np.amin(var),
        'MM_Loss': np.amin(loss),
        'NSW': geo_mean_through_log(var),
        'Total_Var': np.sum(var),
    })

    for i in range(k):
        obj.update({'Loss' + str(i): loss[i], 'Var' + str(i): var[i], 'Best' + str(i): best[i]})

    return obj


def get_recon_error(n, k, d, data, projection):
    """
    Compute per-group average reconstruction error under a linear projection.

    For each group i, this computes the mean squared reconstruction error
    using the rank-d projector P = projection @ projection.T:

        err_i = ||X_i - X_i P||_F^2 / n_i

    Parameters
    ----------
    n : int
        Feature dimension (unused; kept for API consistency).
    k : int
        Number of groups.
    d : int
        Target dimension (unused directly; implied by `projection`).
    data : Sequence[np.ndarray]
        List-like container of length k, where `data[i]` has shape (n_i, n).
    projection : np.ndarray
        Orthonormal basis matrix of shape (n, d). If columns are not orthonormal,
        this computes a generic linear map projection @ projection.T, which is not
        an orthogonal projector.

    Returns
    -------
    list[float]
        List of length k with average (per-sample) squared Frobenius reconstruction
        error for each group.

    Notes
    -----
    This divides by the number of samples in each group (n_i), not by n_i * n.
    """
    err = []
    for i in range(k):
        err.append(
            np.linalg.norm(
                data[i] - np.dot(np.dot(data[i], projection), projection.T),
                'fro'
            ) ** 2 / len(data[i])
        )
    return err


def get_optimal_error(n, k, d, data):
    """
    Compute per-group optimal (PCA) reconstruction error at rank d.

    For each group i, fits PCA on the empirical second-moment/covariance
    matrix (X_i^T X_i / n_i), constructs the rank-d projector onto the top-d
    components, then computes:

        opt_i = ||X_i - X_i P_d||_F^2 / n_i

    Parameters
    ----------
    n : int
        Feature dimension (unused; kept for API consistency).
    k : int
        Number of groups.
    d : int
        Target rank/dimension.
    data : Sequence[np.ndarray]
        List-like container of length k, where `data[i]` has shape (n_i, n).

    Returns
    -------
    list[float]
        List of length k with each group's optimal average squared reconstruction
        error at rank d.

    Dependencies
    ------------
    Requires `std_PCA` from `standard_PCA`, expected to return eigenvectors/loadings
    ordered by decreasing eigenvalue.
    """
    optimal = []
    for i in range(k):
        P_all = std_PCA(data[i].T @ data[i] / len(data[i]), len(data[i][0]))
        p = P_all[:, :d] @ P_all[:, :d].T
        optimal.append(np.linalg.norm(data[i] - np.dot(data[i], p), 'fro')**2 / len(data[i]))
    return optimal


def get_trace(n, k, d, data, projection):
    """
    Compute a per-group trace-based objective under a linear projection.

    For each group i, computes:

        a_i = projection.T @ X_i.T
        tr_i = trace( - (a_i @ X_i @ projection) / n_i )

    Parameters
    ----------
    n : int
        Feature dimension (unused; kept for API consistency).
    k : int
        Number of groups.
    d : int
        Target dimension (unused directly; implied by `projection`).
    data : Sequence[np.ndarray]
        List-like container of length k, where `data[i]` has shape (n_i, n).
    projection : np.ndarray
        Matrix of shape (n, d).

    Returns
    -------
    list[float]
        List of length k containing the computed trace value for each group.

    Notes
    -----
    The negative sign means larger (less negative) values correspond to *smaller*
    trace of (projection.T X_i.T X_i projection) / n_i. If you intended the usual
    PCA "maximize captured variance" trace, you may want to drop the minus sign.
    """
    tr = []
    for i in range(k):
        a = np.dot(projection.T, data[i].T)
        tr.append(np.trace(-np.dot(np.dot(a, data[i]), projection) / len(data[i])))
    return tr


def align_sign(vec1, vec2):
    """
    Align the sign of `vec2` to match `vec1` based on their dot product.

    This is commonly used to make eigenvectors / principal components comparable
    across runs, since eigenvectors are only identifiable up to sign.

    Parameters
    ----------
    vec1 : np.ndarray
        Reference vector of shape (p,) or (p, 1).
    vec2 : np.ndarray
        Vector to be sign-aligned to `vec1`, same shape as `vec1`.

    Returns
    -------
    np.ndarray
        `vec2` possibly multiplied by -1 so that dot(vec1, vec2) >= 0.
    """
    dot_product = np.dot(vec1, vec2)
    if dot_product < 0:
        vec2 = -vec2
    return vec2


def mse_loss_per_row(X, X_hat):
    """
    Compute mean squared error (MSE) per row between two matrices.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features).
    X_hat : np.ndarray
        Reconstructed/predicted array with the same shape as `X`.

    Returns
    -------
    np.ndarray
        1D array of shape (n_samples,) where entry i is the mean squared error
        of row i: mean_j (X[i, j] - X_hat[i, j])^2.

    Raises
    ------
    AssertionError
        If `X.shape != X_hat.shape`.
    """
    assert X.shape == X_hat.shape, "Shapes of X and X_hat must be the same"
    return np.mean((X - X_hat) ** 2, axis=1)
