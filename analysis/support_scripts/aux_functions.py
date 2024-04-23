import pandas as pd
import torch


def df_to_tensor(df):
    return torch.tensor(df.values)

def tensor_to_df(tensor):
    return pd.DataFrame(tensor.numpy())
    