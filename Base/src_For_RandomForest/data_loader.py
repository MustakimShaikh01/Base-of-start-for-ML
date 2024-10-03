import pandas as pd
import json

def load_data(config_file_path):
    # Load configuration
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    # Load train and test datasets
    train_df = pd.read_csv(config['train_file'])
    test_df = pd.read_csv(config['test_file'])
    
    return train_df, test_df, config
