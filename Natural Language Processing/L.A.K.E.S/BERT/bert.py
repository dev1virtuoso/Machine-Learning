import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from datasets import load_dataset

# Load the dataset
ds = load_dataset("nvidia/HelpSteer")
train = ds['train']
val = ds['validation']

# Initialize the model and tokenizer
model_name = "bert-base-uncased"  # Replace with the pre-trained model you choose
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Convert the dataset into the input format suitable for the model
def preprocess(example):
    inputs = tokenizer(example['text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    return inputs.input_ids[0]

train_data = train.map(preprocess)
val_data = val.map(preprocess)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3  # Replace with the desired number of training epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        inputs = batch.to(device)
        labels = inputs.clone()
        labels[labels == tokenizer.mask_token_id] = -100  # Ignore loss for masked tokens

        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits

        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch.to(device)
            labels = inputs.clone()
            labels[labels == tokenizer.mask_token_id] = -100  # Ignore loss for masked tokens

            outputs = model(inputs)
            logits = outputs.logits

            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Print the training and validation losses
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# After training, you can save the model
model.save_pretrained("/path/to/directory")
tokenizer.save_pretrained("/path/to/directory") 
