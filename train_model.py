import pandas as pd
import json


data = json.load(open('LM/article_data.json'))

# Assuming `data` is your loaded JSON data
df = pd.DataFrame(data)

# Convert `created_at` to datetime and extract year, month, etc.
df['created_at'] = pd.to_datetime(df['created'], errors='coerce')
df['year'] = df['created_at'].dt.year
df['month'] = df['created_at'].dt.month

# Drop the original `created_at` column if no longer needed
df.drop(['created_at'], axis=1, inplace=True)


### map reduce step

data = 

###################

from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize `title` + `text`
vectorizer = TfidfVectorizer(max_features=10000)  # Adjust as necessary
text_features = vectorizer.fit_transform(df['title'] + ' ' + df['text']).toarray()

import tensorflow as tf
import numpy as np

# Combine title and text for each record
combined_texts = df['title'] + ' ' + df['text']

# Initialize and fit the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(combined_texts)
total_words = len(tokenizer.word_index) + 1

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(combined_texts)

# Create input sequences and their corresponding labels
input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
X, labels = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)


# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

import matplotlib.pyplot as plt
import time

optimizer = tf.keras.optimizers.Adam(lr=0.01)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Define learning rate schedule
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
model.summary()

# Define the TimeHistory callback
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Initialize the TimeHistory callback
time_callback = TimeHistory()

# Train the model and record the history
history = model.fit(X, y, epochs=100, verbose=1, callbacks=[reduce_lr, time_callback])

# Save the entire model to a HDF5 file
model.save('article_data_model.h5')

# Plotting loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('model_loss.png')

# Plotting accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('model_accuracy.png')

# Plotting time per epoch
plt.figure()
plt.plot(time_callback.times)
plt.title('Time per epoch')
plt.ylabel('Time (seconds)')
plt.xlabel('Epoch')
plt.savefig('time_per_epoch.png')












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