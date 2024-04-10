import tensorflow as tf
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model from the .h5 file
model_path = 'article_data_model_REDUCER.h5'  # Specify the model path
model = tf.keras.models.load_model(model_path)

# Initialize the tokenizer for text processing
# Note: Ensure to use the same preprocessing as was used during the model's training
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Load combined texts data for tokenizer fitting
combined_texts_file = 'all_final_outputs.json'  # Specify the data file
with open(combined_texts_file, 'r') as file:
    combined_texts = json.load(file)

# Fit the tokenizer on the combined texts
tokenizer.fit_on_texts(combined_texts)

# Example input text
input_text = ("we evaluated the states of consciousness of seven persons who had sustained a severe head injury, "
              "and describe the behavioural manifestations associated with four treatment strategies. "
              "the subjects were between the ages of 19 and 55 and were recruited from both acute and")

# Tokenize the input text
input_tokens = tokenizer.texts_to_sequences([input_text])

# Padding parameters
max_sequence_len = 186  # Define the maximum sequence length based on model training
padding_length = max_sequence_len - 1  # Padding length to ensure sequences match model input

# Pad the input tokens to match the model's expected input shape
input_tokens_padded = pad_sequences(input_tokens, maxlen=padding_length, padding='post')

# Generate text based on the input
generated_sequence = []  # Initialize the list to store generated words
current_input = input_tokens_padded  # Start with the initial input sequence

for _ in range(10):  # Generate 10 words
    # Predict the next word's probabilities
    predicted_probs = model.predict(current_input)
    
    # Select the word with the highest probability as the next word
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    predicted_word = tokenizer.index_word.get(predicted_index, '<UNK>')  # Use '<UNK>' for unknown words
    
    # Append the predicted word to the generated sequence
    generated_sequence.append(predicted_word)

    # Update current_input by including the predicted word for the next prediction
    current_input = np.roll(current_input, -1)
    current_input[0, -1] = predicted_index

# Combine the generated words into a single text
generated_text = ' '.join(generated_sequence)
print("Generated Text:", generated_text)
