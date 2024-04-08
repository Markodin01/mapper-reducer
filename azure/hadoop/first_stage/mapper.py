#!/usr/bin/env python3
import sys
import json
import re

def preprocess_text(text):
    """Preprocess a single text string."""
    text = text.lower()  # Convert to lowercase.
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space.
    text = text.replace('\n', ' ')  # Remove new lines.
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters.
    return text

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            # Each line is a raw JSON string
            json_record = json.loads(line)
            # Extract and preprocess the text
            text = preprocess_text(json_record['text'])
            words = text.split()
            # Emit each word count
            for word in words:
                print(f"{word}\t1")
        except ValueError:
            # Handle lines that are not valid JSON
            continue
