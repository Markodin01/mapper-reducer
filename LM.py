import json
import glob
import re
import os

def preprocess_text(text):
    """Preprocess a single text string."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.replace('\n', ' ')  # Remove new lines
    return text

def process_file(file_path):
    """Process a single JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line)
                created = article['created']
                text = article['text']
                # Preprocess the text
                text = preprocess_text(text)

                title, text = extract_title_and_text(text)  # Extract title from text
                # Now you have `created` and `preprocessed text`, you can proceed with your analysis or save the data
                print(created, text, title)  # Example of how to use the data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {file_path}")

def process_folder(folder_path):
    """Process all JSON files in a folder."""
    for file_path in glob.glob(os.path.join(folder_path, '*.json')):
        process_file(file_path)

def extract_title_and_text(text):
    # Regular expression to split the text at the first set of square brackets
    match = re.match(r'\[(.*?)\](.*)', text, re.DOTALL)
    if match:
        title = match.group(1).strip()  # Extracts the title and removes leading/trailing whitespace
        rest_of_text = match.group(2).strip()  # Extracts the rest of the text and removes leading/trailing whitespace
        return title, rest_of_text
    return "", text  # Return the original text if no title is found

# Example usage
folder_path = '/Users/markodin/Desktop/articles/'
process_folder(folder_path)
