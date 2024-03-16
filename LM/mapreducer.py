def mapper(df):
    
    """
    The mapper function takes a text (string) and emits key-value pairs
    for each word in the format (word, 1).
    """
    # Normalize the text to lower case and remove punctuation
    text = df['text'].str.lower()
    text_words = text.str.split().explode()
    
    title = df['title'].str.lower()
    title_text = title.str.split().explode()
    
    list_of_trigrams_for_reducer = []
    list_of_title_words_for_reducer = []
    
    for i in range(len(text_words)-2):
        list_of_trigrams_for_reducer.append(text_words[i]+','+text_words[i+1]+','+text_words[i+2], 1)
        
    for title_word in title_text:
        list_of_title_words_for_reducer.append(title_word, 1)
        
    return list_of_trigrams_for_reducer, list_of_title_words_for_reducer


def reducer_to_dict(mapped_data):
    aggregated_data = {}
    for item in mapped_data:
        key, count = item
        if key in aggregated_data:
            aggregated_data[key] += count
        else:
            aggregated_data[key] = count
    return aggregated_data

import csv

def reducer_to_csv(mapped_data, output_file):
    aggregated_data = reducer_to_dict(mapped_data)  # Reuse the reducer_to_dict function
    
    # Write to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['key', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for key, count in aggregated_data.items():
            writer.writerow({'key': key, 'count': count})