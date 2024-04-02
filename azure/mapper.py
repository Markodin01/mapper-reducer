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
