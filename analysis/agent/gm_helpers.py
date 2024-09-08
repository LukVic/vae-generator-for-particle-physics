import pandas as pd

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
    
    separate_features = {
        'real_features': [],
        'binary_features': [],
        'categorical_features': []
    }
    for feature in df_data.columns:
        unique_num = df_data[feature].nunique()
        if unique_num == 2:
            separate_features['binary_features'].append(feature)
        elif df_data[feature].nunique() < 50:
            separate_features['categorical_features'].append(feature)
        else:
            separate_features['real_features'].append(feature)
    ordered_features = separate_features['real_features'] + separate_features['binary_features'] + separate_features['categorical_features']
    df_data = df_data[ordered_features]
    category_counts = []
    for i, col in enumerate(df_data.columns):

        unique_num = df_data[col].nunique()  
        if unique_num == 2:
            feature_type_dict['binary_param'].append([idx,idx+1])
            feature_type_dict['binary_data'].append(i)
            idx += 1
        elif df_data[col].nunique() < 50:
            feature_type_dict['categorical_param'].append([idx,idx+df_data[col].nunique()])
            feature_type_dict['categorical_data'].append(i)
            idx += df_data[col].nunique()
            category_counts.append(unique_num)
        else: 
            feature_type_dict['real_param'].append([idx,idx+2])
            feature_type_dict['real_data'].append(i)
            idx += 2

    df_cotegorical = df_data.iloc[:, feature_type_dict['categorical_data']]
    df_rest = df_data.iloc[:, feature_type_dict['real_data'] + feature_type_dict['binary_data']]
    df_cotegorical = categorical_one_hot_encode(df_cotegorical, category_counts)
    df_data = pd.concat([df_rest, df_cotegorical.astype(int)], axis=1) 
    feature_type_dict['max_idx'] = idx
    return feature_type_dict, df_data

def categorical_one_hot_encode(df_cat_col, category_counts):
    columns = []
    for idx, feature in enumerate(df_cat_col.columns):
            columns += [f"{feature}_{jdx}" for jdx in range(category_counts[idx])]
    print(columns) 
    df_one_hot = pd.get_dummies(df_cat_col, columns=df_cat_col.columns)
    df_one_hot.columns = columns
    return df_one_hot