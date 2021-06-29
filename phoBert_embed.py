import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def get_embed(text):
    global phobert, tokenizer

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
    with torch.no_grad():
        features = phobert(input_ids)  # Models outputs are now tuples
    pass

if __name__ == '__main__':
    get_embed('Sơn')