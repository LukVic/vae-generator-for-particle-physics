import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd


import time

DEVICE = "cuda"

class Encoder(nn.Module):
    def __init__(self, zdim):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.device = DEVICE

        self.body = nn.Sequential(
            nn.Linear(71, 20),
            nn.ReLU(),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            nn.Linear(20, self.zdim * 2)

            # nn.Linear(71, self.zdim*2)
        )

    def forward(self, x):
        scores = self.body(x.to(self.device))
        mu, sigma = mu, sigma = torch.split(scores, self.zdim, dim=1)
        
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim):
        super(Decoder, self).__init__()
        self.zdim = zdim

        self.body = nn.Sequential(
            nn.Linear(self.zdim, 20),
            nn.ReLU(),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            # nn.Linear(20, 20),
            # nn.ReLU(),
            nn.Linear(20, 71),
            nn.Sigmoid()

            # nn.Linear(self.zdim, 71),
            # nn.Sigmoid()
        )

    def forward(self, z):
        xhat = self.body(z)
        return xhat

class VAE(nn.Module):
    def __init__(self, zdim):
        super(VAE, self).__init__()
        self.device = DEVICE
        self.zdim = zdim
        self.encoder = Encoder(zdim).to(self.device)
        self.decoder = Decoder(zdim).to(self.device)

    def forward(self, x):
        mu, sigma = self.encoder(x.view(-1, 71))
        std = torch.exp(sigma)  
        qz = torch.distributions.Normal(mu, std)
        z = qz.rsample()
        logqz = qz.log_prob(z)
        xhat = self.decoder(z)
        
        return xhat,z,logqz,qz,std

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


    # Computes reconstruction loss
    def recon(self, x_hat, x):
        
        pxz = Normal(x_hat, torch.ones_like(x_hat))
        E_log_pxz = -pxz.log_prob(x.to(self.device)).sum(dim=1)

        return E_log_pxz

    def loss_function(self, x, x_hat, mu, qz, sigma):
        LOSS = self.recon(x_hat, x)
        #eps = 0.00000001
        x_hat = x_hat.to(self.device)
        x = x.to(self.device)
        
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        KLD = torch.distributions.kl_divergence(qz, pz).sum(dim=1)
        beta = 0.3

        return torch.mean(LOSS + beta*KLD)
    

# Define hyperparameters
batch_size = 8
input_size = 71
latent_size = 10
lr = 0.001
num_epochs = 15
elbo_history = []

# Create dataloader
# train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

df = pd.read_csv('/home/lucas/Documents/KYR/msc_thesis/analysis/data/df_tt.csv')
train_dataset = torch.tensor(df.values, dtype=torch.float32)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# Create model and optimizer
model = VAE(latent_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
start_time = time.time()
model.train()
for epoch in range(num_epochs):
    #for i, (x, _) in enumerate(train_dataloader):
    for i, x in enumerate(train_dataloader):
        x = x.view(-1, input_size)
        optimizer.zero_grad()
        x_hat, mu, logvar, qz, std = model(x)
        print(mu)
        loss = model.loss_function(x, x_hat, mu, qz, std)
        loss.backward()
        optimizer.step()
        elbo_history.append(loss.item())
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Generate some samples and their reconstructions
model.eval()
with torch.no_grad():
    x_sample, _ = next(iter(train_dataloader))
    x_sample = x_sample.view(-1, input_size)
    x_hat, _, _,_,_ = model(x_sample)

num_samples = 18
fig, axes = plt.subplots(nrows=6, ncols=6)
udxs = []
for i, ax in enumerate(axes.flat):
    if i < num_samples*2:
        row = i / (6)
        #print(int(row))
        if int(row) % 2 == 0:
            ax.imshow(x_sample[int(row)*6+(i % 6)].view(28, 28), cmap='gray')
        else:
            ax.imshow(x_hat[int(row-1)*6+(i % 6)].to('cpu').view(28, 28), cmap='gray')
        a = int(row)*8+(i % 8)
        b = int(row-1)*8+(i % 8)
        udxs.append(a)
        udxs.append(b)
    ax.axis('off')
idxs = sorted(udxs)
#print(idxs)
plt.show()


# Posterior collapse
with torch.no_grad():
    kl_divs = []
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(DEVICE)
        recon_batch, mu, logvar, qz, std = model(data)
        z = qz.rsample()
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl_div = torch.distributions.kl_divergence(qz, pz)
        kl_divs.append(kl_div)
        break
    
    kl_divs = torch.cat(kl_divs, dim=0)
    kl_divs_mean = kl_divs.mean(dim=0)
    
    # plot histogram of averaged kl divergences for each latent space component
    kl_divs_mean = kl_divs_mean.cpu().numpy()
    plt.figure()
    plt.xlabel('Latent vector component')
    plt.ylabel('Mean KL-divergence accros the batch')
    plt.hist(kl_divs_mean, bins=20)
    #plt.xticks(range(0, 20, 1))
    #plt.savefig('hist_1.png')
    plt.show()
        

# ELBO graph
plt.figure()
plt.plot(elbo_history)
plt.xlabel('Batch number')
plt.ylabel('Total Loss')
plt.title('Total Loss vs. Batch number')
#plt.savefig('col_1.png')
plt.show()

print("Number of VAE params: {0}".format(model.count_params()))