import tensorflow as tf

data = '''
Sentence 1: 今天天氣很好。
Sentence 2: 我打算去公園散步。

Sentence 1: 這部電影真的很有趣。
Sentence 2: 我強烈推薦給你。

Sentence 1: 昨晚的晚餐非常美味。
Sentence 2: 我們去了一家新開的餐廳。

Sentence 1: 運動對身體健康非常重要。
Sentence 2: 每天鍛煉可以增強免疫力。

Sentence 1: 我最喜歡的書是《時間簡史》。
Sentence 2: 這本書闡述了宇宙的起源。'''

sentences = data.split('\n\n')

sentences1 = []
sentences2 = []
for sentence_pair in sentences:
    sentences_split = sentence_pair.split('\n')
    if len(sentences_split) == 2:
        sentence1, sentence2 = sentences_split
        sentences1.append(sentence1.split(':')[1].strip())
        sentences2.append(sentence2.split(':')[1].strip())

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)

sequences1 = tokenizer.texts_to_sequences(sentences1)
sequences2 = tokenizer.texts_to_sequences(sentences2)

max_length = max(max(len(seq1), len(seq2)) for seq1, seq2 in zip(sequences1, sequences2))
padded_sequences1 = tf.keras.preprocessing.sequence.pad_sequences(sequences1, maxlen=max_length, padding='post')
padded_sequences2 = tf.keras.preprocessing.sequence.pad_sequences(sequences2, maxlen=max_length, padding='post')

input_sequences = padded_sequences1
target_sequences = padded_sequences2

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(input_sequences, target_sequences, epochs=10, batch_size=64)

model.save('chinese_language_model_v2.h5')
