from data_loader import load_data
from data_preprocessing import preprocess_data
from model_training import train_model
from prediction import predict
from submission import create_submission
import json

def main():
    # Load configuration from the JSON file
    config_file_path = 'config/config.json'
    
    # Load data
    train_df, test_df, config = load_data(config_file_path)

    # Preprocess the training and test data
    train_df = preprocess_data(train_df, config)
    test_df = preprocess_data(test_df, config)
    
    # Train the model using the training data
    model = train_model(train_df, config['target_column'], config['drop_columns'])
    
    # Make predictions on the test data
    predictions = predict(model, test_df, config['drop_columns'])
    
    # Create a submission file with the predictions
    create_submission(test_df, predictions, 'submission.csv')

if __name__ == "__main__":
    main()
