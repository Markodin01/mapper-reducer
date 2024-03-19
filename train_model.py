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

def mapper(df):
    
    """
    The mapper function takes a text (string) and emits key-value pairs
    for each word in the format (word, 1).
    """
    # Normalize the text to lower case and remove punctuation
    text = df['text'].str.lower()
    text_words = text.str.split().explode().reset_index(drop=True)
    
    title = df['title'].str.lower()
    title_text = title.str.split().explode()
    
    list_of_trigrams_for_reducer = []
    list_of_title_words_for_reducer = []
    
    for i in range(len(text_words)):
        list_of_trigrams_for_reducer.append((text_words[i], 1))
        
    for title_word in title_text:
        list_of_title_words_for_reducer.append((title_word, 1))
        
    return list_of_trigrams_for_reducer, list_of_title_words_for_reducer


def reducer_to_dict(mapped_data):
    aggregated_data_title = {}
    aggregated_data_text = {}
    
    # Process title words
    for title_word in mapped_data[0]:  # Assuming mapped_data[0] is the list of title words
        key = title_word[0]  # The key is the word itself
        count = title_word[1]
        if key in aggregated_data_title:
            aggregated_data_title[key] += count
        else:
            aggregated_data_title[key] = count
    
    # Process trigrams
    for trigram in mapped_data[1]:  # Assuming mapped_data[1] is the list of trigrams
        # Sort the words in the trigram to make the order irrelevant
        words_sorted = ','.join(sorted(trigram[0].split(',')))
        count = trigram[1]
        if words_sorted in aggregated_data_text:
            aggregated_data_text[words_sorted] += count
        else:
            aggregated_data_text[words_sorted] = count
    
    return aggregated_data_title, aggregated_data_text




def reducer_to_df(mapped_data):
    """
    Takes the mapped data, aggregates it using a reducer function to create two dictionaries,
    and then converts those dictionaries into pandas DataFrames.
    """
    aggregated_data_title, aggregated_data_text = reducer_to_dict(mapped_data)  # reducer_to_dict returns two dictionaries
    
    # Convert the aggregated data to lists of dictionaries suitable for DataFrame creation
    data_for_df_title = [{'key': key, 'count': count} for key, count in aggregated_data_title.items()]
    data_for_df_text = [{'key': key, 'count': count} for key, count in aggregated_data_text.items()]
    
    # Create the DataFrames
    df_title = pd.DataFrame(data_for_df_title)
    df_text = pd.DataFrame(data_for_df_text)
    
    return df_title, df_text

df_text, df_title = reducer_to_df(mapper(df))

###############################################

import tensorflow as tf
import numpy as np

# Add 'type' column to both dataframes
df_text['type'] = 'text'
df_title['type'] = 'title'

# Concatenate the dataframes vertically
df_combined = pd.concat([df_text, df_title], ignore_index=True)

# Define the pattern to match
pattern = '[.,()]}{'

# Filter out rows where 'key' contains the pattern
df_combined = df_combined[~df_combined['key'].str.contains(pattern)]

# Initialize and fit the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_combined)
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











