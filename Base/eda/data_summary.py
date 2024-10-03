import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def summarize_data(df):
    """Generate a summary of the DataFrame."""
    summary = {
        'Shape': df.shape,
        'Data Types': df.dtypes.to_dict(),
        'Missing Values': df.isnull().sum().to_dict(),
        'Basic Statistics': df.describe(include='all').to_dict()
    }
    return summary

def run_summary(file_path):
    """Load data and run summary."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}.")
        
        summary = summarize_data(df)
        logging.info(f"Data Summary:\n{summary}")
        return summary
        
    except Exception as e:
        logging.error(f"Error loading data or generating summary: {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/target/titanic/train/train.csv'  # Example path for standalone testing
    run_summary(file_path)
