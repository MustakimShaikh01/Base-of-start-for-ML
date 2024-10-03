def predict(model, test_df, drop_columns):
    X_test = test_df.drop(columns=drop_columns)
    predictions = model.predict(X_test)
    return predictions
