import torch

def angle_to_tan(features_idx, data):
    data_T = data.T
    for idx in features_idx:
        data_T[idx] = torch.tan(data_T[idx])
    data = data_T.T
    print(data.shape)
    return data

def tan_to_angle(features_idx, data):
    data_T = data.T
    for idx in features_idx:
        data_T[idx] = torch.atan(data_T[idx])
    data = data_T.T
    return data