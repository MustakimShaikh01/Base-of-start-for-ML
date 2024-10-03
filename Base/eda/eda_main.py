import os
import logging
import sys
from data_summary import run_summary
from missing_values_analysis import run_missing_values_analysis
from distribution_visualization import run_distribution_visualization
from correlation_analysis import run_correlation_analysis
from categorical_analysis import run_categorical_analysis

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report(project_folder, summary, missing_values, distributions, correlations, categorical_features):
    """Generate a markdown report and save it."""
    report_path = os.path.join(project_folder, 'eda_results', 'report.md')
    
    with open(report_path, 'w') as report_file:
        report_file.write('# EDA Report\n\n')
        
        # Summary section
        report_file.write('## Summary\n')
        report_file.write(f'- **Shape**: {summary["Shape"]}\n')
        report_file.write(f'- **Data Types**:\n```\n{summary["Data Types"]}\n```\n')
        report_file.write(f'- **Missing Values**:\n```\n{summary["Missing Values"]}\n```\n')
        report_file.write(f'- **Basic Statistics**:\n```\n{summary["Basic Statistics"]}\n```\n\n')

        # Missing Values section
        report_file.write('## Missing Values\n')
        report_file.write('The following columns have missing values:\n')
        for col, count in missing_values.items():
            report_file.write(f'- **{col}**: {count} missing values\n')
        report_file.write('\n')  # New line for better separation
        
        # Distributions section
        report_file.write('## Distributions\n')
        for col, plot_path in distributions.items():
            report_file.write(f'### {col}\n')
            report_file.write(f'![Distribution of {col}]({plot_path})\n\n')

        # Correlation section
        report_file.write('## Correlation Matrix\n')
        report_file.write('![Correlation Matrix](correlation_matrix.png)\n\n')
        
        # Categorical Features section
        report_file.write('## Categorical Features\n')
        for col, plot_path in categorical_features.items():
            report_file.write(f'### {col}\n')
            report_file.write(f'![Count Plot of {col}]({plot_path})\n\n')
    
    logging.info(f"Report generated and saved to {report_path}")

def run_all_eda(project_folder):
    """Run all EDA scripts on the training dataset."""
    
    # Create eda_results directory if it doesn't exist
    os.makedirs(os.path.join(project_folder, 'eda_results'), exist_ok=True)
    
    train_file = os.path.join(project_folder, 'train', 'train.csv')  # Adjust based on your CSV file structure
    
    summary = run_summary(train_file)
    missing_values = run_missing_values_analysis(train_file)
    distributions = run_distribution_visualization(train_file)
    correlations = run_correlation_analysis(train_file)
    categorical_features = run_categorical_analysis(train_file)

    # Generate report with all the collected information
    generate_report(project_folder, summary, missing_values or {}, distributions or {}, correlations or {}, categorical_features or {})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python eda/eda_main.py <project_folder>")
        sys.exit(1)

    # Get the project folder from command line arguments
    project_folder = sys.argv[1]
    
    if not os.path.exists(project_folder):
        logging.error(f"The specified project folder does not exist: {project_folder}")
        sys.exit(1)

    logging.info("Starting EDA process.")
    run_all_eda(project_folder)
    logging.info("EDA process completed.")
