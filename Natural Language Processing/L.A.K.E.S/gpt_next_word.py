import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Get user input from the terminal
text = input("Enter a text: ")

# Encode the user input
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens to a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the Dropout modules
model.eval()

# If you have a GPU, put everything on CUDA
if torch.cuda.is_available():
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')

# Set the number of threads to use all available CPU cores
torch.set_num_threads(torch.get_num_threads())

# Move tokens_tensor back to CPU if running on GPU
if tokens_tensor.is_cuda:
    tokens_tensor = tokens_tensor.to('cpu')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Get the predicted next sub-word
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# Print the predicted word
print(predicted_text)
