import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import AG_NEWS
from torchtext.data import Field, LabelField, BucketIterator

# Define the CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        output = self.fc(cat)
        return output

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
vocab_size = 25000
embedding_dim = 100
num_filters = 100
filter_sizes = [3, 4, 5]
output_dim = 4
dropout = 0.5
batch_size = 64
epochs = 10

# Define the fields
text_field = Field(lower=True, tokenize='spacy')
label_field = LabelField(dtype=torch.float)

# Load the AG_NEWS dataset
train_data, test_data = AG_NEWS.splits(text_field, label_field)

# Build the vocabulary
text_field.build_vocab(train_data, max_size=vocab_size)

# Create the iterators
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device
)

# Initialize the CNN model
model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout).to(device)

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    
    for batch in train_iterator:
        optimizer.zero_grad()
        text = batch.text
        labels = batch.label
        output = model(text).squeeze(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
    
    train_loss = total_loss / len(train_iterator)
    train_acc = total_correct / len(train_data)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
