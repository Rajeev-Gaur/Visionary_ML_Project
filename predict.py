import joblib

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, input_data):
    return model.predict(input_data)
