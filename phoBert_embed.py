import torch
import joblib
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

try:
    embed_vocab = joblib.load('models/embed_vocab.pkl')
except:
    embed_vocab = None


def get_word_embed(word):
    global phobert, tokenizer

    input_ids = torch.tensor([tokenizer.encode(word)])
    with torch.no_grad():
        features = phobert(input_ids)  # Models outputs are now tuples
    cls = features[0][0][0]  # cls token
    return cls


def build_sample_tensor(input_tensor, device='cuda'):
    global embed_vocab
    result = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 768))
    for i, row in enumerate(input_tensor):
        for j, idx in enumerate(row):
            emb = embed_vocab[idx.item()]
            result[i, j] = emb
    return result.to(device)


def get_vocab_embed(vocab):
    global embed_vocab
    if embed_vocab is None:
        embed_vocab = {}
        for i, w in vocab.itos.items():
            v = get_word_embed(w)
            embed_vocab.update({i: v})
            print(f'\rgot vocab word embedding of token {i + 1}th...', end='', flush=True)
        print('')
        joblib.dump(embed_vocab, 'models/embed_vocab.pkl')


if __name__ == '__main__':
    x = torch.LongTensor([[100, 200, 5]])
    build_sample_tensor(x)
