import pandas as pd

def preprocess_data(data):
    # Forward fill missing values
    data.ffill(inplace=True)
    
    # Create a target variable if it doesn't exist
    if 'target' not in data.columns:
        data['target'] = data['Close']  # Use 'Close' as the target variable
    
    # One-hot encoding for categorical variables
    data = pd.get_dummies(data, drop_first=True)  # drop_first to avoid dummy variable trap
    
    print("Processed columns:", data.columns)  # Check columns after preprocessing
    return data

