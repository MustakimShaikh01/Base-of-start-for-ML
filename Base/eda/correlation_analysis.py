import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_correlation_matrix(df):
    """Plot the correlation matrix and save the figure."""
    plt.figure(figsize=(10, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('eda_results/correlation_matrix.png')  # Save the figure
    plt.close()  # Close the figure to prevent display

def run_correlation_analysis(file_path):
    """Load data and analyze correlations."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}.")
        
        plot_correlation_matrix(df)
        return df.corr().to_dict()  # Return correlation matrix as dictionary
        
    except Exception as e:
        logging.error(f"Error loading data or plotting correlations: {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/target/titanic/train/train.csv'  # Example path for standalone testing
    run_correlation_analysis(file_path)
