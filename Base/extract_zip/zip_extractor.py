import os
import zipfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory(path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Created directory: {path}")
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")

def move_csv_files(source_folder, project_folder):
    """Move CSV files to appropriate folders."""
    train_folder = os.path.join(project_folder, 'train')
    test_folder = os.path.join(project_folder, 'test')
    extra_folder = os.path.join(project_folder, 'extra')

    # Create necessary folders
    create_directory(train_folder)
    create_directory(test_folder)
    create_directory(extra_folder)

    # Move CSV files to their respective folders
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            source_path = os.path.join(source_folder, file_name)

            # Determine destination based on file name
            if 'train' in file_name.lower():
                dest_path = os.path.join(train_folder, file_name)
            elif 'test' in file_name.lower():
                dest_path = os.path.join(test_folder, file_name)
            else:
                dest_path = os.path.join(extra_folder, file_name)

            # Avoid overwriting existing files
            if not os.path.exists(dest_path):
                try:
                    os.rename(source_path, dest_path)
                    logging.info(f"Moved {file_name} to {os.path.basename(dest_path)}")
                except Exception as e:
                    logging.error(f"Error moving {file_name} to {dest_path}: {e}")
            else:
                logging.warning(f"File already exists: {dest_path}. Skipping move.")

def process_zip_file(source_folder, data_folder):
    """Process the latest ZIP file and organize its contents."""
    # Get all ZIP files
    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]
    
    if not zip_files:
        logging.warning("No ZIP files found in the source folder.")
        return

    # Find the latest ZIP file
    try:
        latest_zip_file = max(zip_files, key=lambda f: os.path.getctime(os.path.join(source_folder, f)))
        latest_zip_path = os.path.join(source_folder, latest_zip_file)

        # Create project folder based on the ZIP file name
        project_name = os.path.splitext(latest_zip_file)[0]
        project_folder = os.path.join(data_folder, project_name)

        create_directory(project_folder)

        # Extract the ZIP file
        with zipfile.ZipFile(latest_zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_folder)
            logging.info(f"Extracted {latest_zip_file} to {project_folder}")

        # Move CSV files from the project folder
        move_csv_files(project_folder, project_folder)

        logging.info(f"Processed {latest_zip_file} into {project_folder} with train, test, and extra folders.")
    
    except zipfile.BadZipFile:
        logging.error(f"{latest_zip_file} is not a valid ZIP file.")
    except Exception as e:
        logging.error(f"Error processing {latest_zip_file}: {e}")
