import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_categorical_features(df):
    """Plot count plots for categorical features and save the figures."""
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.savefig(f'eda_results/categorical_{col}.png')  # Save the figure
        plt.close()  # Close the figure to prevent display

def run_categorical_analysis(file_path):
    """Load data and analyze categorical features."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}.")
        
        plot_categorical_features(df)
        return df.select_dtypes(include='object').columns.tolist()  # Return list of categorical columns
        
    except Exception as e:
        logging.error(f"Error loading data or plotting categorical features: {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/target/titanic/train/train.csv'  # Example path for standalone testing
    run_categorical_analysis(file_path)
