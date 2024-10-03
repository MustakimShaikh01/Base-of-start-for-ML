import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_missing_values(df):
    """Visualize missing values in the DataFrame."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('eda_results/missing_values_heatmap.png')  # Save the figure
    plt.close()  # Close the figure to prevent display

def run_missing_values_analysis(file_path):
    """Load data and analyze missing values."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}.")
        
        plot_missing_values(df)
        missing_values_info = df.isnull().sum().to_dict()
        return missing_values_info
        
    except Exception as e:
        logging.error(f"Error loading data or plotting missing values: {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/target/titanic/train/train.csv'  # Example path for standalone testing
    run_missing_values_analysis(file_path)
