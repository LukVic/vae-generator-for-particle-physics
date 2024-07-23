import csv
import json

def load_features(path_features: str, file_features: str) -> list:
    """Reads features from the csv file

    Args:
        path_features (str): general features path
        file_features (str): specific features file

    Returns:
        list: list of loaded features
    """
    features = []
    # read features from the csv file
    with open(f'{path_features}{file_features}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append(row[0])
    return features

def load_config(path_conf: str) -> dict:
    """load config file

    Args:
        path_conf (str): config path

    Returns:
        dict: json to python dictionary
    """
    with open(f"{path_conf}hyperparams.json", 'r') as json_file:
        conf_dict = json.load(json_file)
    return conf_dict