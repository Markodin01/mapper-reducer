# mapper.py

import numpy as np
import pickle

def load_model(model_path):
    # Load the initial model from a shared location
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def save_model(model, model_path):
    # Save the updated model to a shared location
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

def mapper(_, value):
    # Load the initial model
    model_path = 's3://your-model-bucket/model.pkl'
    model = load_model(model_path)

    # Assume each key-value pair represents a data point with features and label
    features = np.array(value[:-1], dtype=float)
    label = int(value[-1])

    # Update the model parameters based on the received data point
    # Custom machine learning algorithm logic for parameter updates
    model.update(features, label)

    # Save the updated model to a shared location for Reducer to retrieve
    save_model(model, model_path)

    # Emit intermediate results (optional)
    yield ('training_progress', 1)
