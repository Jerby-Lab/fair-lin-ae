import torch 
import torch.nn.functional as F


class CVaR_loss(torch.nn.Module):
    """
    Conditional Value-at-Risk (CVaR) surrogate loss. 

    This loss implements a mini-batch approximation to CVaR optimization,
    focusing training on the worst-performing fraction of samples according to
    a surrogate per-sample reconstruction loss (default: MSE).

    Parameters
    ----------
    beta : float, default=0.2
        Tail fraction defining the CVaR level. Roughly corresponds to optimizing
        the worst `beta * 100%%` of samples in each minibatch.
    eta : float, default=0.01
        Step size for updating the internal threshold parameter `u`, which
        estimates the (1 − beta) quantile of the loss distribution.
    surr_loss : {"MSE"}, default="MSE"
        Type of per-sample surrogate loss to use. Currently only mean squared
        error is implemented.
    device : torch.device or None, optional
        Device on which internal state (e.g., `u`) is stored. If None, defaults
        to CUDA when available, otherwise CPU.
    """

    def __init__(self,
                 beta=0.2,
                 eta=0.01,
                 surr_loss='MSE',
                 device=None):
        super(CVaR_loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.beta = beta
        self.eta = eta
        self.u = torch.tensor([0.0]).to(self.device).detach()
        if surr_loss == 'MSE':
            self.surrogate_loss = torch.nn.MSELoss(reduce=False)
        else:
            assert surr_loss+' is not implemnted'


    def forward(self, y_pred, y_true):
        batch_size = y_true.shape[0] 
        surr_loss = self.surrogate_loss(y_pred, y_true).mean(dim=-1, keepdim=True)
        p = surr_loss > self.u
        self.u = self.u-self.eta*(1 - p.sum().detach()/(self.beta*batch_size))
        p.detach_()
        loss = torch.mean(p * surr_loss) / self.beta
        return loss

class KL_loss(torch.nn.Module):
    """
    KL-divergence-based distributionally robust loss

    This loss implements a KL-DRO objective by exponentially reweighting
    per-sample surrogate losses, yielding robustness to high-loss (outlier)
    samples while maintaining smooth optimization behavior.

    Parameters
    ----------
    Lambda : float, default=1.0
        Robustness parameter controlling the strength of reweighting.
        Smaller values concentrate weight more heavily on high-loss samples.
    gamma : float, default=0.9
        Exponential moving-average parameter for stabilizing the normalization
        constant `u`. When gamma = 1.0, no moving average is applied.
    surr_loss : {"MSE"}, default="MSE"
        Type of per-sample surrogate loss to use. Currently only mean squared
        error is implemented.
    device : torch.device or None, optional
        Device on which internal state (e.g., `u`, `prev_max`) is stored. If None,
        defaults to CUDA when available, otherwise CPU.
    """
    def __init__(self,
                 Lambda=1.0,
                 gamma=0.9, # when gamma=1.0, no moving average applied
                 surr_loss='MSE',
                 device=None):
        super(KL_loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.Lambda = Lambda
        self.gamma = gamma
        self.u = torch.tensor([0.0]).to(self.device).detach()
        self.prev_max = torch.tensor([0.0]).to(self.device).detach()
        if surr_loss == 'MSE':
            self.surrogate_loss = torch.nn.MSELoss(reduce=False)
        else:
            assert surr_loss+' is not implemnted'


    def forward(self, y_pred, y_true):
        surr_loss = self.surrogate_loss(y_pred, y_true).mean(dim=-1, keepdim=True)
        max_loss = torch.max(surr_loss).detach()
        surr_loss -= max_loss 
        exp_loss = torch.exp(surr_loss/self.Lambda)
        self.u = (1 - self.gamma) * self.u * torch.exp(torch.clip((self.prev_max-max_loss)/self.Lambda, max=10)) + self.gamma * (exp_loss.mean().detach())
        #self.u = (exp_loss.mean().detach())
        p = exp_loss/self.u
        p.detach_()
        loss = torch.mean(p * surr_loss) + self.Lambda*(p * p.log()).sum()
        self.prev_max = max_loss
        return loss

class fair_kl_loss(torch.nn.Module):
    """
    Fairness-aware KL-divergence DRO loss across multiple data groups.

    This loss extends KL-based distributionally robust optimization to a
    group-aware setting by reweighting group-level excess reconstruction errors
    relative to group-specific optimal baselines.

    Parameters
    ----------
    Lambda : float, default=1.0
        Robustness parameter controlling how strongly the objective emphasizes
        groups with higher excess reconstruction error.
    gamma : float, default=0.9
        Unused smoothing parameter retained for API consistency with other
        KL-based losses.
    device : torch.device or None, optional
        Device on which tensors are stored. If None, defaults to CUDA when
        available, otherwise CPU.
    data : list of array-like
        List of group-specific data matrices. Each element is converted to a
        torch tensor of shape (n_i, n_features).
    op_error : list of array-like
        List of group-specific optimal reconstruction errors (one per group),
        used as baselines when computing excess error.

    Returns
    -------
    loss : torch.Tensor
        Scalar fairness-aware KL-DRO objective aggregating group-level losses.
    """
    def __init__(self, Lambda=1.0, gamma=0.9, device=None, data=None, op_error=None):
        super(fair_kl_loss, self).__init__()
        self.Lambda = Lambda
        self.gamma = gamma
        self.data = [torch.tensor(d).to(device) for d in data]
        self.op_error = [torch.tensor(op).to(device) for op in op_error]
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.u = torch.tensor([0.0]).to(self.device).detach()
        self.prev_max = torch.tensor([0.0]).to(self.device).detach()

    def forward(self, U, y_pred, y_true):
        errors = []
        for Xi in self.data:
            Xi_hat = torch.matmul(Xi, torch.matmul(U, U.T))
            errors.append(torch.mean((Xi - Xi_hat) ** 2, dim=1))

        losses_i = []
        for err, op_err in zip(errors, self.op_error):
            losses_i.append((err - op_err) / len(op_err))
        
        losses = torch.stack(losses_i)
        
        max_l_i = torch.max(losses).detach()
        
        # Apply the re-weighting scheme
        exp_losses = torch.exp((losses - max_l_i) / self.Lambda)
        
        p = exp_losses / exp_losses.sum(dim=0, keepdim=True)
        
        weighted_loss = torch.mean(p * losses) + self.Lambda * (p * p.log()).sum()
        
        return weighted_loss
