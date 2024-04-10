from pyspark.sql import SparkSession
import json

spark = SparkSession.builder.appName("LargeResultSize").config("spark.driver.maxResultSize", "2g").getOrCreate()

df = spark.read.option("multiLine", "true").json("LM/article_data.json")
df.show()


from collections import defaultdict

def batch_process_reducer(batch_mapped_data, min_threshold=5, max_threshold=100):
    """
    Reducer function that processes batches of mapped data, aggregates counts,
    and then filters based on the frequency thresholds.
    
    Parameters:
    - batch_mapped_data (list of tuples): Mapped data containing word-count pairs.
    - min_threshold (int): Minimum frequency threshold for words to be kept.
    - max_threshold (int): Maximum frequency threshold for words to be kept.
    
    Returns:
    - dict: A dictionary of words and their counts, filtered based on the thresholds.
    """
    local_word_counts = defaultdict(int)

    # Aggregate word counts for the batch
    for word, count in batch_mapped_data:
        local_word_counts[word] += count

    # Apply filtering within the reducer for this batch
    words_to_filter = {word: count for word, count in local_word_counts.items() if min_threshold <= count <= max_threshold}
    
    return words_to_filter

def batch_process_filter_out(batch_filtered_counts, processed_text):
    """
    Filters out words from a json_record's text that are not in the batch_filtered_counts.

    Parameters:
    - batch_filtered_counts (dict): A dictionary of words to keep, based on previous filtering.
    - json_record (dict): A dictionary representing a single JSON record, expected to contain a 'text' key.

    Returns:
    - tuple: The filtered text string and the percentage reduction in word count.
    """
    original_words = processed_text.split()
    original_word_count = len(original_words)

    filtered_words = [word for word in original_words if word in batch_filtered_counts]
    filtered_word_count = len(filtered_words)

    reduction_percentage = ((original_word_count - filtered_word_count) / original_word_count) * 100 if original_word_count > 0 else 0

    return ' '.join(filtered_words), reduction_percentage

import re

def preprocess_text(text):
    """
    Preprocess a single text string by converting to lowercase, replacing multiple
    spaces with a single space, removing new lines, and stripping non-alphanumeric characters.

    Parameters:
    - text (str): The text string to preprocess.

    Returns:
    - str: The preprocessed text.
    """
    text = text.lower()  # Convert to lowercase.
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space.
    text = text.replace('\n', ' ')  # Remove new lines.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters.
    return text


def mapper(json_record):
    """
    Process a single JSON record to preprocess the text contained within and map
    each word to a count of 1, essentially tokenizing and counting the occurrence of each word.

    Parameters:
    - json_record (dict): A dictionary representing a single JSON record, expected to contain a 'text' key.

    Returns:
    - list of tuples: A list where each tuple contains a word and the count 1.
    """
    text = preprocess_text(json_record['text'])
    words = text.split()
    return [(word, 1) for word in words]


batch_mapped_data = [mapper(record) for record in data]
flattened_mapped_data = [item for sublist in batch_mapped_data for item in sublist]
batch_filtered_counts = batch_process_reducer(flattened_mapped_data)
outputs_and_reductions = [batch_process_filter_out(batch_filtered_counts, record) for record in data]

final_outputs, percentage_reductions = zip(*outputs_and_reductions)
