import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_names = ['EleutherAI/pythia-2.8B-deduped',
               'bigscience/bloom-3b',
               'cerebras/Cerebras-GPT-2.7B']  # bigger, may overflow RAM
model_name = model_names[2]
print('Making string tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'Loading model `{model_name}`...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # automatically store parameters on gpu, cpu or disk
    low_cpu_mem_usage=True,  # try to limit RAM
    load_in_8bit=True,  # load model in low precision to save memory
    torch_dtype=torch.float16,  # load model in low precision to save memory
    offload_state_dict=True,  # offload onto disk if needed
    offload_folder="offload",  # offload model to `offload/`
)
print('Finished loading model')
