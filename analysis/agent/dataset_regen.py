import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS):
    
    print(f'{PATH_MODEL}{DATA_FILE}_{EPOCHS}.pth')
    model = torch.load(f'{PATH_MODEL}{DATA_FILE}_{EPOCHS}.pth')
    df_real = pd.read_csv(f'{PATH_DATA}{DATA_FILE}.csv')
    
    train_dataset = torch.tensor(df_real.values, dtype=torch.float32)
    input_size = train_dataset.shape[1]
    scaler = StandardScaler()
    train_dataset_norm = scaler.fit_transform(train_dataset)
    
    model.eval()
    
    data_array = np.empty((0, input_size), dtype=np.float32)
    batch_size = 64
    with torch.no_grad():
        for idx in range(0, len(train_dataset_norm), batch_size):
            print(f"Regenerating event number: {idx}")
            batch_x = train_dataset_norm[idx : idx + batch_size]
            x_hat_batch, _, _ = model(torch.tensor(batch_x).float())
            x_hat_denorm_batch = scaler.inverse_transform(x_hat_batch.view(-1, input_size).cpu().numpy())
            data_array = np.vstack((data_array, x_hat_denorm_batch))

    # Create a DataFrame from the NumPy array
    df_regen = pd.DataFrame(data_array,columns=df_real.columns)

    print("Processing completed.")
        
    # with torch.no_grad():
    #     for idx, x in enumerate(train_dataset_norm):
    #         print(f"Regenerating event number: {idx}")
    #         x_hat, _, _ = model(torch.tensor(x).float())
    #         #x_denorm = scaler.inverse_transform(x.clone().detach().view(-1,input_size).to('cpu'))[0]
    #         x_hat_denorm = scaler.inverse_transform(x_hat.clone().detach().view(-1,input_size).to('cpu'))[0]
    #         #print(f"This is the real sample: {x_denorm}")
    #         #print(f"This is the generated sample: {x_hat_denorm}")
            
    #         df_regen = pd.concat([df_regen,pd.DataFrame([x_hat_denorm])], ignore_index=True)

    df_regen.to_csv(f'{PATH_DATA}{DATA_FILE}_regen_{EPOCHS}.csv', index=False)

def main():
    PATH_MODEL = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/models/'
    PATH_DATA = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/data/'
    DATA_FILE = 'df_low'
    EPOCHS = 20
    
    dataset_regen(PATH_DATA, DATA_FILE, PATH_MODEL, EPOCHS)
    
if __name__ == "__main__":
    main()