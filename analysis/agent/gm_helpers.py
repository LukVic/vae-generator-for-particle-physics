import pandas as pd

def infer_feature_type(df_data, PATH_FEATURES):
    idx, jdx, kdx = 0, 0, 0
    feature_type_dict = {
        'binary_param': [],
        'categorical_param': [],
        'real_param': [],
        'max_idx': None,
        'binary_data': [],
        'categorical_data': [],
        'real_data': [],
        'categorical_one_hot': [],
        'categorical_only': []
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
    df_data_one_hot = df_data.copy()
    
    # Open a file in write mode
    with open(f"{PATH_FEATURES}.csv", "w") as file:
        # Write each string to a new line
        for string in ordered_features:
            file.write(string + "\n")
    
    category_counts = []
    for i, col in enumerate(df_data_one_hot.columns):

        unique_num = df_data_one_hot[col].nunique()  
        if unique_num == 2:
            feature_type_dict['binary_param'].append([idx,idx+1])
            feature_type_dict['binary_data'].append(i)
            idx += 1
            jdx += 1
        elif df_data_one_hot[col].nunique() < 50:
            feature_type_dict['categorical_param'].append([idx,idx+df_data_one_hot[col].nunique()])
            feature_type_dict['categorical_one_hot'].append([jdx,jdx+df_data_one_hot[col].nunique()])
            feature_type_dict['categorical_only'].append([kdx,kdx+df_data_one_hot[col].nunique()])
            feature_type_dict['categorical_data'].append(i)
            category_counts.append(unique_num)
            idx += df_data_one_hot[col].nunique()
            jdx += df_data_one_hot[col].nunique()
            kdx += df_data_one_hot[col].nunique()
        else: 
            feature_type_dict['real_param'].append([idx,idx+2])
            feature_type_dict['real_data'].append(i)
            idx += 2
            jdx += 1
    df_cotegorical = df_data_one_hot.iloc[:, feature_type_dict['categorical_data']]
    df_rest = df_data_one_hot.iloc[:, feature_type_dict['real_data'] + feature_type_dict['binary_data']]
    df_cotegorical = categorical_one_hot_encode(df_cotegorical, category_counts)
    df_data_one_hot = pd.concat([df_rest, df_cotegorical.astype(int)], axis=1) 
    feature_type_dict['max_idx'] = idx
    return feature_type_dict, df_data, df_data_one_hot

def categorical_one_hot_encode(df_cat_col, category_counts):
    columns = []
    for idx, feature in enumerate(df_cat_col.columns):
            columns += [f"{feature}_{jdx}" for jdx in range(category_counts[idx])]
    #print(columns) 
    df_one_hot = pd.get_dummies(df_cat_col, columns=df_cat_col.columns)
    df_one_hot.columns = columns
    return df_one_hot