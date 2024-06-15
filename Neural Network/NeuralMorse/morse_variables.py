input_vocab = sorted(set(''.join(input_messages) + '0123456789'))
target_vocab = sorted(set(' '.join(target_messages).split()))

input_token_index = dict([(ch, i) for i, ch in enumerate(input_vocab)])
for ch, code in morse_code.items():
    input_vocab.append(ch)
    input_token_index[ch] = len(input_vocab) - 1

target_token_index = dict([(ch, i) for i, ch in enumerate(target_vocab)])

num_encoder_tokens = len(input_vocab)
num_decoder_tokens = len(target_vocab)
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in target_sequences])
