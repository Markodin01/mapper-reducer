# reducer.py

import numpy as np
import pickle

def load_model(model_path):
    # Load the latest model from a shared location
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def save_model(model, model_path):
    # Save the aggregated model to a shared location
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

def reducer(_, values):
    # Initialize model parameters
    model_path = 's3://your-model-bucket/model.pkl'
    model = load_model(model_path)

    # Iterate over received training progress signals
    for _ in values:
        pass  # Optional: Perform any logic needed for training progress tracking

    # Save the aggregated model to a shared location
    save_model(model, model_path)

    # Emit the final model parameters (optional)
    yield ('final_model_parameters', model.get_parameters())
