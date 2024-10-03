import pandas as pd

def create_submission(test_df, predictions, output_file):
    # Create a DataFrame for the submission
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],  # Ensure PassengerId is taken from the test DataFrame
        'Survived': predictions  # The predictions from your model
    })
    
    # Save the submission DataFrame to a CSV file
    submission.to_csv(output_file, index=False)  # Save without the index column
    print(f'Submission file created: {output_file}')
