import torch
import joblib
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
try:
    vocab = joblib.load('models/trg_vocab.pkl')
except:
    vocab = None

def get_word_embed(text):
    global phobert, tokenizer

    input_ids = torch.tensor([tokenizer.encode(text)])
    with torch.no_grad():
        features = phobert(input_ids)  # Models outputs are now tuples
    cls = features[0][0][0] # cls token
    return cls

def build_sample_tensor(input_tensor, device='cuda'):
    global vocab
    if vocab is None:
        vocab = joblib.load('models/trg_vocab.pkl')
    result = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 768))
    for i, row in enumerate(input_tensor):
        for j, idx in enumerate(row):
            text = vocab.itos[idx.item()]
            result[i, j] = get_word_embed(text)
    return result.to(device)

if __name__ == '__main__':
    x = torch.LongTensor([[100, 200, 5]])
    build_sample_tensor(x)