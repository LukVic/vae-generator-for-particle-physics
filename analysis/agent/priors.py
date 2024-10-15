import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class StandardPrior(nn.Module):
    def __init__(self, L=2):
        super(StandardPrior, self).__init__()
        self.L = L 
        self.means = torch.zeros(1, L)
        self.logvars = torch.zeros(1, L)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.L)
    
    def log_prob(self, z):
        return self.log_standard_normal(z)

    def log_standard_normal(self, x, reduction=None, dim=None):
        log_p = -0.5 * torch.log(2 * torch.tensor(torch.pi)) - 0.5 * x**2
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p.sum(dim=1)

# class MoGPrior(nn.Module):
#     def __init__(self, L, num_components, device):
#         super(MoGPrior, self).__init__()
#         multiplier = 1
#         self.device = device
#         self.L = L
#         self.num_components = num_components

#         self.means = nn.Parameter(torch.randn(num_components, self.L, device=device) * multiplier)
#         self.logvars = nn.Parameter(torch.randn(num_components, self.L, device=device))
#         self.w = nn.Parameter(torch.zeros(num_components, 1, 1, device=device))

#         # Precompute constant terms
#         self.log_2pi = torch.log(2 * torch.tensor(torch.pi, device=device))

#     def get_params(self):
#         return self.means, self.logvars

#     def sample(self, batch_size):
#         means, logvars = self.get_params()
#         w = F.softmax(self.w, dim=0).squeeze()
        
#         # Ensure w is on the correct device
#         w = w.to(self.device)
        
#         # Use w.device to ensure indexes are on the same device as w
#         indexes = torch.multinomial(w, batch_size, replacement=True)
        
#         eps = torch.randn(batch_size, self.L, device=self.device)
#         z = means[indexes] + eps * torch.exp(0.5 * logvars[indexes])
#         return z

#     def log_prob(self, z):
#         means, logvars = self.get_params()
#         w = F.softmax(self.w, dim=0)
        
#         z = z.unsqueeze(0).to(self.device)
#         means = means.unsqueeze(1)
#         logvars = logvars.unsqueeze(1)

#         log_p = self.log_normal_diag(z, means, logvars) + torch.log(w)
#         log_prob = torch.logsumexp(log_p, dim=0, keepdim=False)
#         return log_prob.T

#     def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
#         log_p = -0.5 * (self.log_2pi + log_var + torch.exp(-log_var) * (x - mu)**2)
#         if reduction == 'avg':
#             return torch.mean(log_p, dim)
#         elif reduction == 'sum':
#             return torch.sum(log_p, dim)
#         else:
#             return log_p

class MoGPrior(nn.Module):
    def __init__(self, L, num_components, device):
        super(MoGPrior, self).__init__()
        multiplier = 1
        self.device = device
        self.L = L
        self.num_components = num_components

        # params
        self.means = nn.Parameter(torch.randn(num_components, self.L)*multiplier).to(self.device)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L)).to(self.device)

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1)).to(self.device)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze().to(self.device)

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L, device=self.device)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        means = means.unsqueeze(1) # K x 1 x L
        logvars = logvars.unsqueeze(1) # K x 1 x L

        log_p = self.log_normal_diag(z, means, logvars) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob.T
    
    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        PI = torch.from_numpy(np.asarray(np.pi))
        log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p


class FlowPrior(nn.Module):
    def __init__(self, nets, nett, num_flows, D=2, device=None):
        super(FlowPrior, self).__init__()

        self.device = device

        self.D = D

        self.t = nn.ModuleList([nett().to(self.device) for _ in range(num_flows)])
        self.s = nn.ModuleList([nets().to(self.device) for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        xa = xa.to(self.device)
        xb = xb.to(self.device)  # Ensure xb is also on the correct device
        s = self.s[index](xa).to(self.device)
        t = self.t[index](xa).to(self.device)

        if forward:
            # yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            # xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1).to(self.device)  # Ensure permuted tensor is on the correct device

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]).to(self.device), x.to(self.device)
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        x = z.to(self.device)
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.D).to(self.device)
        x = self.f_inv(z)
        return x.view(-1, self.D)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = self.log_standard_normal(z) + log_det_J.unsqueeze(1)
        return log_p

    def log_standard_normal(self, x):
        # Implementing the log probability for a standard normal distribution
        log_p = -0.5 * (x**2 + np.log(2 * np.pi))
        return log_p.sum(dim=1)
        #log_p = -0.5 * torch.log(2 * torch.tensor(torch.pi)) - 0.5 * x**2
    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        PI = torch.tensor(np.pi).to(self.device)  # Ensure constant is on the correct device
        log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p