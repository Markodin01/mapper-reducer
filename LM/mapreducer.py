data = json.load(open('LM/article_data.json'))

import json
import re
from collections import defaultdict

def mapper(json_record):
    """Preprocess the text and map words to counts."""
    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = text.replace('\n', ' ')  # Remove new lines
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
        return text

    # Extract and preprocess the text
    text = json_record['text']
    text = preprocess_text(text)
    
    words = text.split()
    return [(word, 1) for word in words]


def batch_process_reducer(batch_mapped_data, min_threshold=5, max_threshold=100):
    """
    Reducer function that processes batches of mapped data, aggregates counts,
    and then filters based on the frequency thresholds.
    """
    local_word_counts = defaultdict(int)
    # Aggregate word counts for the batch
    for word_count in batch_mapped_data:
        for word, count in word_count:
            local_word_counts[word] += count

    # Apply filtering within the reducer for this batch
    words_to_filter = {word: count for word, count in local_word_counts.items() if min_threshold <= count <= max_threshold}
    
    return words_to_filter

    
def batch_process_filter_out(batch_filtered_counts, json_record):
    """
    Filters out words from json_record['text'] that are not in batch_filtered_counts.
    Returns the filtered text and the percentage reduction in word count.
    """
    # Normalize the text: lowercase and remove non-alphanumeric characters
    text = json_record['text'].lower()
    text = re.sub(r'\W+', ' ', text)
    
    # Tokenize the normalized text into words
    original_words = text.split()
    original_word_count = len(original_words)
    
    # Filter the words based on batch_filtered_counts
    filtered_words = [word for word in original_words if word in batch_filtered_counts]
    filtered_word_count = len(filtered_words)
    
    # Calculate the percentage reduction
    if original_word_count > 0:
        reduction_percentage = ((original_word_count - filtered_word_count) / original_word_count) * 100
    else:
        reduction_percentage = 0  # To handle cases with empty text
    
    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text, reduction_percentage


    

# Load data
with open('LM/article_data.json', 'r') as file:
    data = json.load(file)
    
all_final_outputs = []  # To store all filtered texts from all batches
all_percentage_reductions = []  # To store percentage reductions from all batches

# Example: Batch processing
batch_size = 25
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_mapped_data = [mapper(record) for record in batch]
    # Flatten the list of lists into a single list
    flattened_mapped_data = [item for sublist in batch_mapped_data for item in sublist]
    batch_filtered_counts = batch_process_reducer(batch_mapped_data)
    outputs_and_reductions = [batch_process_filter_out(batch_filtered_counts, record) for record in batch]
    final_outputs, percentage_reductions = zip(*outputs_and_reductions)
    # Now, batch_filtered_counts holds the filtered word counts for this batch
    # You can process these counts as needed (e.g., accumulate them, analyze, store, etc.)
    all_final_outputs.append(final_outputs)
    all_percentage_reductions.append(percentage_reductions)
    

###############################################

# flatten the reuslts
all_final_outputs = [item for sublist in all_final_outputs for item in sublist]
all_percentage_reductions = [item for sublist in all_percentage_reductions for item in sublist]