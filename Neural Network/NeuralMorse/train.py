import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

morse_code = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
    'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
    'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.'
}

input_messages = ['HELLO', 'WORLD', 'OPENAI']
target_messages = ['.... . .-.. .-.. ---', '.-- --- .-. .-.. -..', '--- .--. . -. .-.-.-']

input_vocab = sorted(set(''.join(input_messages) + '0123456789'))
target_vocab = sorted(set(' '.join(target_messages).split()))

input_token_index = dict([(ch, i) for i, ch in enumerate(input_vocab)])
for ch, code in morse_code.items():
    input_vocab.append(ch)
    input_token_index[ch] = len(input_vocab) - 1

target_token_index = dict([(ch, i) for i, ch in enumerate(target_vocab)])

num_encoder_tokens = len(input_vocab)
num_decoder_tokens = len(target_vocab)
max_encoder_seq_length = max([len(seq) for seq in input_messages])
max_decoder_seq_length = max([len(seq.split()) for seq in target_messages])
target_vocab.append('<end>')
target_token_index['<end>'] = len(target_vocab) - 1
num_decoder_tokens = len(target_vocab)

# Convert text data to numerical sequences
encoder_input_data = np.zeros(
    (len(input_messages), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_messages), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_messages), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_messages, target_messages)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0

    for t, char in enumerate(target_text.split()):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# Define the neural network model
latent_dim = 256
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=50, validation_split=0.2)

# Save the model
model.save('morse_translation_model.h5')
