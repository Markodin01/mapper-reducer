import numpy as np
import pickle
import logging
import os

def load_model(model_path):
    """
    Load the machine learning model from the specified path.

    Args:
    model_path (str): Path to the model file.

    Returns:
    object: The loaded model, or None if loading failed.
    """
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        return None  # Consider initializing a new model here if appropriate

def save_model(model, model_path):
    """
    Save the machine learning model to the specified path.

    Args:
    model (object): The model to be saved.
    model_path (str): Path to save the model file.
    """
    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def mapper(_, value):
    """
    Process each data point to update the machine learning model.

    The function assumes that each value in the data is a list where the last element is the label
    and the remaining elements are features.

    Args:
    _ (ignore): Ignored key from the key-value pair in MapReduce. Often unused in ML tasks.
    value (list): The value in the key-value pair, representing a data point.

    Yields:
    tuple: A tuple indicating progress in training.
    """
    model_path = 's3://your-model-bucket/model.pkl'

    # Validate model path
    if not os.path.exists(model_path):
        logging.error("Model path does not exist.")
        return  # Early exit if model path is invalid

    model = load_model(model_path)
    if model is None:
        return  # Exit if model loading failed

    # Process input data
    try:
        features = np.array(value[:-1], dtype=float)
        label = int(value[-1])
    except Exception as e:
        logging.error(f"Error processing input data: {e}")
        return

    # Update model with new data point
    model.update(features, label)

    # Save the updated model
    save_model(model, model_path)

    # Emit a signal indicating a data point has been processed
    yield ('training_progress', 1)
