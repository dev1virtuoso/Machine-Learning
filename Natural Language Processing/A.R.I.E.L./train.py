import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torchtext.data.metrics import bleu_score
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set random seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Define input and output fields
tokenizer = T5Tokenizer.from_pretrained('t5-base')
SRC = Field(tokenize=tokenizer.tokenize, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenizer.tokenize, init_token='<sos>', eos_token='<eos>', lower=True)

# Load and split the dataset
train_data, valid_data, test_data = TranslationDataset.splits(
    path='data_path', exts=('.src', '.trg'), fields=(SRC, TRG))

# Build the vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, labels=trg)
        loss = criterion(output.logits.view(-1, output.logits.shape[-1]), trg.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Define the evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg

            output = model(src, labels=trg)
            loss = criterion(output.logits.view(-1, output.logits.shape[-1]), trg.view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Train the model
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

# Load the saved best model
model.load_state_dict(torch.load('model.pt'))

# Test the model
test_loss = evaluate(model, test_iterator, criterion)

# Perform inference with the model
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    tokens = src_field.tokenize(sentence)
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids=src_tensor, max_length=max_len, num_beams=5)
    
    trg_text = trg_field.decode(output[0], skip_special_tokens=True)
    return trg_text

# Example usage
example_sentence = "How are you?"
translation = translate_sentence(example_sentence, SRC, TRG, model, device)
print(f'Source: {example_sentence}')
print(f'Translation: {translation}')
