def infer_feature_type(df_data):
    idx = 0
    
    feature_type_dict = {
        'binary_param': [],
        'categorical_param': [],
        'real_param': [],
        'max_idx': None,
        'binary_data': [],
        'categorical_data': [],
        'real_data': []
    }
    
    for i, col in enumerate(df_data.columns):
        unique_num = df_data[col].nunique()  
        if unique_num == 2:
            feature_type_dict['binary_param'].append([idx,idx+1])
            feature_type_dict['binary_data'].append(i)
            idx += 1
        elif df_data[col].nunique() < 50:
            feature_type_dict['categorical_param'].append([idx,idx+df_data[col]])
            feature_type_dict['categorical_data'].append(i)
            idx += df_data[col]
        else: 
            feature_type_dict['real_param'].append([idx,idx+2])
            feature_type_dict['real_data'].append(i)
            idx += 2
    
    feature_type_dict['max_idx'] = idx
    return feature_type_dict

def categorical_one_hot_encode(df_cat_col):
    df_one_hot = df_cat_col.get_dummies(df_cat_col, columns=['Category'])
    print(df_one_hot)
