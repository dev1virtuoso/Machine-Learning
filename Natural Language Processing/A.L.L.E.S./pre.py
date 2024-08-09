import tensorflow as tf
import numpy as np

data_path = '/path/to/directory'


with open(data_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text_data])
sequences = tokenizer.texts_to_sequences([text_data])
sequences = np.array(sequences)

input_sequences = sequences[:, :-1]
target_sequences = sequences[:, 1:]

dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))

batch_size = 64
buffer_size = 10000
dataset = dataset.shuffle(buffer_size).batch(batch_size)

output_dir = '/path/to/directory'
np.savetxt(output_dir + 'input_sequences.txt', input_sequences, delimiter=',', fmt='%d')
np.savetxt(output_dir + 'target_sequences.txt', target_sequences, delimiter=',', fmt='%d')
