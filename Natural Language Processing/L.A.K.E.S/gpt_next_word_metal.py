import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.onnx as onnx
import coremltools as ct

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text input
text = "What is the fastest car in the"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens to a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the Dropout modules
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# Get the predicted next sub-word
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# Print the predicted word
print(predicted_text)

# Convert PyTorch model to ONNX format
dummy_input = tokens_tensor.to('cpu')
onnx_path = "gpt2_model.onnx"
onnx.export(model, dummy_input, onnx_path)

# Convert ONNX model to Core ML model with Metal GPU acceleration
mlmodel_path = "gpt2_model.mlmodel"
mlmodel = ct.converters.onnx.convert(onnx_path, minimum_ios_deployment_target='13', use_float_arraytype=True)
mlmodel.save(mlmodel_path)
