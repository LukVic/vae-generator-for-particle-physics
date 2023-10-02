import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.normal import Normal


class Encoder(nn.Module):
    def __init__(self, zdim, input_size):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size

        self.body_con = nn.Sequential(
            nn.Linear(self.input_size[0], 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, self.zdim*2)

            # nn.Linear(72, self.zdim*2)
        )
        
        self.body_cat = nn.Sequential(
            nn.Linear(self.input_size[1], 120),
            nn.ReLU(),
            nn.Linear(120, 200),
            nn.ReLU(),
            nn.Linear(200, 120),
            nn.ReLU(),
            nn.Linear(120, self.zdim*self.input_size[1])

            # nn.Linear(72, self.zdim*2)
        )

    def forward(self, x):
        scores_con = self.body_con(x[0].to('cuda:0'))
        logits_cat = self.body_cat(x[1].to('cuda:0'))
        mu_con, sigma_con = torch.split(scores_con, self.zdim, dim=1)
        return mu_con, sigma_con, logits_cat.view(-1, self.zdim, self.input_size[1])

class Decoder(nn.Module):
    def __init__(self, zdim, input_size):
        super(Decoder, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        
        self.body_con = nn.Sequential(
            nn.Linear(self.zdim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, self.input_size[0]),
            nn.Sigmoid()

            # nn.Linear(self.zdim, 72),
            # nn.Sigmoid()
        )
        
        self.body_cat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size[1]*self.zdim, 120),
            nn.ReLU(),
            nn.Linear(120, 200),
            nn.ReLU(),
            nn.Linear(200, 120),
            nn.ReLU(),
            nn.Linear(120,  self.input_size[1]),
            nn.Sigmoid()

            # nn.Linear(self.zdim, 72),
            # nn.Sigmoid()
        )

    def forward(self, z):
        xhat_con = self.body_con(z[0])
        xhat_cat = self.body_cat(z[1])
        return [xhat_con, xhat_cat.view(-1, self.input_size[1])]

class VAE(nn.Module):
    def __init__(self, zdim, input_size):
        super(VAE, self).__init__()
        self.device = 'cuda:0'
        self.zdim = zdim
        self.input_size = input_size
        self.encoder = Encoder(self.zdim, self.input_size).to('cuda:0')
        self.decoder = Decoder(self.zdim, self.input_size).to('cuda:0')

    def forward(self, x):
        temperature = 1.0
        mu_con, sigma_con, phi_cat = self.encoder(x)
        
        std_con = torch.exp(sigma_con)  
        
        qz_con = torch.distributions.Normal(mu_con, std_con)
        pz_con = torch.distributions.Normal(torch.zeros_like(mu_con), torch.ones_like(sigma_con))
        zs_con = qz_con.rsample()
        zs_cat = self.gumbel_softmax(phi_cat, temperature, batch=True)
        x_hats = self.decoder([zs_con, zs_cat])
        #?logqz = qz.log_prob(z)

        
        return x_hats, pz_con, qz_con, phi_cat


    def loss_function(self, xs, x_hats, pz, qz, phi):

        
        #============================CONTINUOUS======================================================
        
        ELBO_con = self.recon_con(x_hats[0], xs[0])
        #eps = 0.00000001
        x_hats[0] = x_hats[0].to('cuda:0')
        xs[0] = xs[0].to('cuda:0')
        #BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 72), reduction='sum')
        #BCE = -torch.sum(x * torch.log(x_hat + eps) + (1 - x) * torch.log(1 - x_hat + eps))
        #KLD = torch.distributions.kl_divergence(qz, pz).sum()
        KLD_con = torch.distributions.kl_divergence(qz, pz).sum(dim=1)
        # if i == 0:
        #     beta = 0.0
        # elif i == 1:
        #     beta = 0.1
        # else: beta = 0.3
        #if i % 2 == 0: beta = 0.2
        #else: beta = 0.7
        beta = 1
        
        #=============================CATEGORICAL==================================================
        
        ELBO_cat = (torch.nn.functional.binary_cross_entropy(x_hats[1].to('cuda:0'), xs[1].to('cuda:0'), reduction="none").sum()) / xs[1].to('cuda:0').shape[0]
        KLD_cat = torch.mean(torch.sum(self.kl_cat(phi.to('cuda:0')), dim=1))


        LOSS_con = ELBO_con + beta*KLD_con
        LOSS_cat = ELBO_cat + beta*KLD_cat
        # print("SEPARATE LOSS")
        # print(f"CONTINUOUS LOSS: {torch.mean(LOSS_con)}")
        # print(f"CATEGORICAL LOSS: {torch.mean(LOSS_cat)}")
        # print(f"CONTINUOUS KLD: {torch.mean(KLD_con)}")
        # print(f"CATEGORICAL KLD: {torch.mean(KLD_cat)}")
        
        return torch.mean(LOSS_con)
    

    def gumbel_distribution_sample(self, shape: torch.Size, eps=1e-20) -> torch.Tensor:
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_distribution_sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        y = logits + self.gumbel_distribution_sample(logits.shape).to('cuda:0')
        return torch.nn.functional.softmax(y / temperature, dim=-1)
    
    
    def gumbel_softmax(self, logits, temperature, batch):
        input_shape = logits.shape
        if batch: 
            b, n, k = input_shape
            logits = logits.view(b*n, k)
        y = self.gumbel_softmax_distribution_sample(logits, temperature)    
        #? n_classes = input_shape[-1]
        #? print(n_classes)
        return y.view(input_shape)
        
    def kl_cat(self, phi: torch.Tensor) -> torch.Tensor:
        # phi is logits of shape [B, N, K] where B is batch, N is number of categorical distributions, K is number of classes
        B, N, K = phi.shape
        phi = phi.view(B*N, K)
        q = dist.Categorical(logits=phi)
        p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K).to('cuda:0')) # uniform bunch of K-class categorical distributions
        kl = dist.kl.kl_divergence(q, p) # kl is of shape [B*N]
        return kl.view(B, N)       

    # Computes reconstruction loss
    def recon_con(self, x_hat, x):
        pxz = Normal(x_hat, torch.ones_like(x_hat))
        E_log_pxz = -pxz.log_prob(x.to('cuda:0')).sum(dim=1)

        return E_log_pxz
    
    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)