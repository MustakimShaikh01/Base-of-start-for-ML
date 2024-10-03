import pandas as pd

def preprocess_data(df, config):
    # Handle missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['float64', 'int64']:
                df[column] = df[column].fillna(df[column].median())  # Assign back to the DataFrame
            else:
                df[column] = df[column].fillna(df[column].mode()[0])  # Assign back to the DataFrame

    # Create new features (example: FamilySize)
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Encode categorical variables
    df = pd.get_dummies(df, columns=config['categorical_columns'], drop_first=True)

    # Drop unwanted columns (this line is now commented out)
    # df.drop(columns=config['drop_columns'], inplace=True)  # Drop unwanted columns (remove this line)

    return df
