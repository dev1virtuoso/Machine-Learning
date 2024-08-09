import numpy as np

input_file = '/path/to/directory'
target_file = '/path/to/directory'

with open(input_file, 'r') as f:
    input_sequences = f.readlines()

with open(target_file, 'r') as f:
    target_sequences = f.readlines()

input_sequences = [seq.strip() for seq in input_sequences]

target_sequences = [seq.strip() for seq in target_sequences]

print('Example input sequence:', input_sequences[0])
print('Example of target sequence:', target_sequences[0])
