import tensorflow as tf
import numpy as np

# REDUCED TEXTS
combined_texts = all_final_outputs
mode = 'REDUCER'

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
model.save(f'article_data_model_{mode}.h5')

# Plotting loss
plt.figure()
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(f'model_loss{mode}.png')

# Plotting accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig(f'model_accuracy{mode}.png')

# Plotting time per epoch
plt.figure()
plt.plot(time_callback.times)
plt.title('Time per epoch')
plt.ylabel('Time (seconds)')
plt.xlabel('Epoch')
plt.savefig(f'time_per_epoch{mode}.png')