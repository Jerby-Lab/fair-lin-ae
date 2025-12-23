import torch 
import torch.nn.functional as F


class CVaR_loss(torch.nn.Module):
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

# class fair_kl_loss(torch.nn.Module):
#     def __init__(self, Lambda=1.0, gamma=0.9, device=None, 
#                  data = None,
#                 op_error = None):
#         super(fair_kl_loss, self).__init__()
#         self.Lambda = Lambda
#         self.gamma = gamma
#         self.data = data
#         self.op_error = None
#         if not device:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
#         self.u = torch.tensor([0.0]).to(self.device).detach()
#         self.prev_max = torch.tensor([0.0]).to(self.device).detach()

#     def forward(self, U, y_pred, y_true):
#         # Compute projections
#         surr_loss = self.surrogate_loss(y_pred, y_true).mean(dim=-1, keepdim=True)
#         max_loss = torch.max(surr_loss).detach()
#         surr_loss -= max_loss 
        
#         errors = []
#         for Xi in self.data: 
#             Xi_hat = Xi @ U @ U.T
#             errors.append(np.mean((X - X_hat) ** 2, axis=1))
#         errors.detach()

#         losses_i = []
#         for err, op_err in zip(errors, self.op_err):
#             losses_i.append((err - op_err) / len(self.op_err)
#         losses_i.detatch()          
        
#         max_l_i = max(losses_i).detach()
#         losses = torch.stack([losses_i])
        
#         # Apply the re-weighting scheme
#         exp_losses = torch.exp((losses - max_l_i) / self.Lambda)
        
#         self.u = (1 - self.gamma) * self.u * torch.exp(torch.clip((self.prev_max - max_loss) / self.Lambda, max=10)) + self.gamma * exp_losses.mean().detach()
        
#         p = exp_losses / self.u
#         p.detach_()
#         weighted_loss = torch.mean(p * losses) + self.Lambda * (p * p.log()).sum()
        
#         self.prev_max = max_loss
        
#         return weighted_loss

class fair_kl_loss(torch.nn.Module):
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
