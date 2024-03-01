import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def preprocess_file(file_path):
    """Function to preprocess an individual stock file."""
    df = pd.read_csv(file_path)
    # Example preprocessing steps:
    # - Select relevant columns
    # - Perform any necessary cleaning or transformations
    # Return the preprocessed DataFrame
    return df

def process_batch(file_paths):
    """Function to process a batch of files."""
    batch_dfs = []
    for file_path in file_paths:
        df = preprocess_file(file_path)
        batch_dfs.append(df)
    # Combine DataFrames from the same batch if necessary
    # Example: return pd.concat(batch_dfs)
    return batch_dfs

def batch_files(file_paths, batch_size):
    """Yield successive n-sized chunks from file_paths."""
    for i in range(0, len(file_paths), batch_size):
        yield file_paths[i:i + batch_size]

# Assuming stock_data_dir is the directory containing your stock files
stock_data_dir = 'path/to/your/stock_data'
file_paths = [os.path.join(stock_data_dir, f) for f in os.listdir(stock_data_dir) if f.endswith('.csv')]

# Define batch size - the number of files to process in parallel
batch_size = 10  # Adjust based on your system's capabilities

# Create batches of file paths
file_batches = list(batch_files(file_paths, batch_size))

# Process batches in parallel
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_batch, file_batches))

# At this point, 'results' contains the processed data from all files
# You might need to further combine or aggregate these results based on your needs
