import pandas as pd
import json


data = json.load(open('LM/article_data.json'))

import json
import re
from collections import defaultdict

def mapper(json_record):
    """
    Mapper function to process each record.
    """
    text = json_record['text'].lower()
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    return [(word, 1) for word in words]

def batch_process_reducer(batch_mapped_data, min_threshold=5, max_threshold=1000):
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
    filtered_word_counts = {word: count for word, count in local_word_counts.items() if min_threshold <= count <= max_threshold}
    
    return filtered_word_counts

# Load data
with open('LM/article_data.json', 'r') as file:
    data = json.load(file)

# Example: Batch processing
batch_size = 25
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_mapped_data = [mapper(record) for record in batch]
    # Flatten the list of lists into a single list
    flattened_mapped_data = [item for sublist in batch_mapped_data for item in sublist]
    batch_filtered_counts = batch_process_reducer(batch_mapped_data)
    # Now, batch_filtered_counts holds the filtered word counts for this batch
    # You can process these counts as needed (e.g., accumulate them, analyze, store, etc.)


###############################################

import tensorflow as tf
import numpy as np


# Initialize and fit the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_combined[['key','type']])
total_words = len(tokenizer.word_index) + 1

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(df_combined['key'])

# Create input sequences and their corresponding labels
input_sequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))







from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model

# Text input
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = Embedding(input_dim=total_words, output_dim=64)(text_input)
encoded_text = LSTM(32)(embedded_text)

# Count input
count_input = Input(shape=(1,), name='count')

# Concatenate the outputs of the two branches
concatenated = Concatenate()([encoded_text, count_input])

# Add a dense layer
output = Dense(1, activation='sigmoid')(concatenated)

# Instantiate the model
model = Model([text_input, count_input], output)







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











