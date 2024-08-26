def infer_feature_type(df_data):
    idx = 0
    
    feature_type_dict = {
        'binary': [],
        'categorical': [],
        'real': [],
        'max_idx': None
    }
    
    for col in df_data.columns:
        unique_num = df_data[col].nunique()  
        if unique_num == 2:
            feature_type_dict['binary'].append([idx,idx+1])
            idx += 1
        elif df_data[col].nunique() < 50:
            feature_type_dict['categorical'].append([idx,idx+df_data[col]])
            idx += df_data[col]
        else: 
            feature_type_dict['real'].append([idx,idx+2])
            idx += 2
    
    feature_type_dict['max_idx'] = idx
    #print(feature_type_dict)
    return feature_type_dict
