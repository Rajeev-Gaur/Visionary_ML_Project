from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y):
      model = Ridge(alpha=1.0)   # Example model
      model.fit(X, y)
      return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) 
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    
    return y_pred 
