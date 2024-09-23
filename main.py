import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score
import config
from data_loader import load_data  
from preprocess_data import preprocess_data
from train_model import train_model
from predict import load_model, make_predictions 
from plot_utils import plot_predictions
import matplotlib.pyplot as plt
import seaborn as sns

# Define output directory
output_dir = 'output'
# Create directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    # Step 1: Load data
    data = load_data(config.DATA_PATH)
    print("Loaded data columns:", data.columns)

    # Step 2: Preprocess data
    processed_data = preprocess_data(data)
    print("Processed data columns:", processed_data.columns)

    # Step 3: Check if 'target' exists and assign target variable
    if 'target' not in processed_data.columns:
        print("Warning: 'target' column not found in processed data.")
        # Assuming you want to predict 'Close' as the target
        target_column = 'Close'  # Change this based on your needs
    else:
        target_column = 'target'

    # Step 4: Split data
    X = processed_data.drop(target_column, axis=1)
    y = processed_data[target_column]

    # Step 5: Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    model = train_model(X_train, y_train)

    # Step 6: Save the trained model
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    
    # Step 7: Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation results
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Step 8: Visualize results
    plot_predictions(y_test, y_pred)
    
    # Optional: Save evaluation results to a text file
    with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"R^2 Score: {r2}\n")
    
if __name__ == "__main__":
    main()


