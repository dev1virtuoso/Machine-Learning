import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from datasets import load_dataset
from multiprocessing import Pool

# Parallel loading of the dataset
def load_helpsteer_dataset():
    return load_dataset("nvidia/HelpSteer")

# Check example structure
def check_example_structure(dataset):
    example = dataset[0]
    print(example)

# Load the dataset
train = load_helpsteer_dataset()['train']

# Call the function with your dataset
check_example_structure(train)
