import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    print(f'{PATH_MODEL}{DATA_FILE}_{EPOCHS}.pth')
    model = torch.load(f'{PATH_MODEL}{DATA_FILE}_disc_{EPOCHS}_sym.pth')
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    input_size = train_dataset.shape[1]
    scaler = StandardScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    
    model.eval()
    latent_dimension = 100
    
    data_array = np.empty((0, input_size), dtype=np.float32)

    with torch.no_grad():
        latent_samples = torch.randn(train_dataset.shape[0], latent_dimension)
        xhats_mu_gauss, xhats_sigma_gauss, xhats_bernoulli = model.decoder(latent_samples.to('cuda'))
        px_gauss = torch.distributions.Normal(xhats_mu_gauss, torch.exp(xhats_sigma_gauss))
        xhat_gauss = px_gauss.sample()
        xhat_bernoulli = torch.bernoulli(xhats_bernoulli)
        xhats = torch.cat((xhat_gauss, xhat_bernoulli.view(-1,1)), dim=1)
        x_hats_denorm = scaler.inverse_transform(xhats.cpu().numpy())
        data_array = np.vstack((data_array, x_hats_denorm))

    # Create a DataFrame from the NumPy array
    df_regen = pd.DataFrame(data_array,columns=df_real.columns)

    print("Processing completed.")

    df_regen.to_csv(f'{PATH_DATA}{DATA_FILE}_disc_{EPOCHS}_new_sym.csv', index=False)

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    DATA_FILE = 'df_no_zeros'
    EPOCHS = 1000
    
    dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    
if __name__ == "__main__":
    main()