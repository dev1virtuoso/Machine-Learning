import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

model = load_model('morse_translation_model.h5')

encoder_model = Model(inputs=model.input[0], outputs=model.get_layer(index=6).output)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.get_layer(index=3)
decoder_outputs, state_h, state_c = decoder_lstm(
    model.input[1], initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_dense = model.get_layer(index=4)
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [model.input[1]] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
input_seq[0, 0, input_token_index['H']] = 1.0
input_seq[0, 1, input_token_index['E']] = 1.0
input_seq[0, 2, input_token_index['L']] = 1.0
input_seq[0, 3, input_token_index['L']] = 1.0
input_seq[0, 4, input_token_index['O']] = 1.0

encoder_outputs, state_h_enc, state_c_enc = encoder_model.predict(input_seq)

target_seq = np.zeros((1, 1, num_decoder_tokens))
target_seq[0, 0, target_token_index['<start>']] = 1.0

output_seq = []
for _ in range(max_decoder_seq_length):
    output_tokens, h, c = decoder_model.predict([target_seq] + [state_h_enc, state_c_enc])

    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = target_vocab[sampled_token_index]
    output_seq.append(sampled_char)

    if sampled_char == '<end>' or len(output_seq) >= max_decoder_seq_length:
        break

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.0

    state_h_enc, state_c_enc = h, c

output_text = ' '.join(output_seq)
print(output_text)
