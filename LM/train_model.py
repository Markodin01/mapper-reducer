import pandas as pd
import json


data = json.load(open('LM/article_data.json'))

# Assuming `data` is your loaded JSON data
df = pd.DataFrame(data)

# Convert `created_at` to datetime and extract year, month, etc.
df['created_at'] = pd.to_datetime(df['created'], errors='coerce')
df['year'] = df['created_at'].dt.year
df['month'] = df['created_at'].dt.month
# Add more temporal features as needed

# Drop the original `created_at` column if no longer needed
df.drop(['created_at'], axis=1, inplace=True)



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
model.add(tf.keras.layers.LSTM(150, return_sequences=True))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=100, verbose=1)

