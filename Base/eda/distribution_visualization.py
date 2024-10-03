import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_feature_distributions(df):
    """Plot distributions of numeric features and save the figures."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'eda_results/distribution_{col}.png')  # Save the figure
        plt.close()  # Close the figure to prevent display

def run_distribution_visualization(file_path):
    """Load data and plot distributions."""
    try:
        numeric_cols = df.select_dtypes(include='number').columns
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}.")
        
        plot_feature_distributions(df)
        return numeric_cols.tolist()  # Return the list of numeric columns
        
    except Exception as e:
        logging.error(f"Error loading data or plotting distributions: {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/target/titanic/train/train.csv'  # Example path for standalone testing
    run_distribution_visualization(file_path)
