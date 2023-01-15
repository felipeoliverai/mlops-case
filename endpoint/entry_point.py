import tarfile
import joblib
import pandas as pd

def model_fn(model_dir):
    # Extract the tar.gz archive
    with tarfile.open(model_dir, mode='r:gz') as archive:
        archive.extractall()
    # Load the model file
    model = joblib.load('model.joblib')
    return model

def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        input_data = pd.read_csv(input_data)
    return input_data

def predict_fn(inputs, model):
    return model.predict(inputs)

def output_fn(prediction, content_type):
    return prediction
