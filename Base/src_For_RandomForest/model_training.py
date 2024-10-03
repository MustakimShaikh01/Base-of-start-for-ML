from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(train_df, target_column, drop_columns):
    # Exclude the target column and unwanted columns from the feature set
    X = train_df.drop(columns=[target_column] + drop_columns)  # Drop target and unwanted columns
    y = train_df[target_column]  # Define the target variable

    # Split the dataset into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate and print the accuracy of the model on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {accuracy:.4f}')  # Print accuracy with four decimal places
    
    return model  # Return the trained model
