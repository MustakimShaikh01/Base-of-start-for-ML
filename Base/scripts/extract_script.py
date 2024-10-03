import os
import sys
import logging

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjust the system path to include the extract_zip directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'extract_zip')))

from extract_zip.zip_extractor import process_zip_file

if __name__ == "__main__":
    # Define the source folder where your ZIP files are stored
    source_folder = os.path.join('data', 'source')
    
    # Define the base data folder where projects will be dynamically created
    data_folder = os.path.join('data', 'target')

    # Log the start of the process
    logging.info("Starting the ZIP file processing.")

    # Call the function to process the latest zip file dynamically
    process_zip_file(source_folder, data_folder)

    # Log the end of the process
    logging.info("ZIP file processing completed.")


# $env:PYTHONPATH="C:\Users\user\Desktop\ML - Code\Base"
# python scripts\extract_script.py