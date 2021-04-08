import re
import spacy
import numpy as np
import joblib as pickle
from transformers import *

from vocabulary import Vocabulary

import html

src_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

max_vocab_size_trg = 15000
max_seq_len_src = 75
max_seq_len_trg = 75
min_freq = 3


def remove_punc(words):
    result = map(lambda w: re.sub('[,.!;:\"\'?<>{}\[\]()-]', '', w), words)
    result = map(lambda w: re.sub('(\d+,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)', 'number', w), result)
    result = list(filter(lambda w: len(w) > 0, result))
    return result

def load_data_from_file(data_file, build_vocab=False, min_freq=1, max_vocab_size=5000):
    print(f'loading data from {data_file} ...')
    with open(data_file) as fp:
        data = []
        for i, text in enumerate(fp):
            data.append(html.unescape(text.strip()))
            print(f'\rprocessed {i+1} sentences ...', end='', flush=True)
        print('')
        data = [remove_punc(tok.split()) for tok in data]
        if build_vocab:
            vocab = Vocabulary()
            vocab.build_vocab(data, lower=True, min_freq=min_freq,
                              max_vocab_size=max_vocab_size)
            return data, vocab
        else:
            return data

def encode_trg_data(data, vocab, max_seq_len):
    result = []
    for s in data:
        ss = [vocab.bos_idx]
        for w in s:
            try:
                idx = vocab.stoi[w.lower()]
            except:
                idx = vocab.unk_idx
            ss.append(idx)
        ss.append(vocab.eos_idx)
        if len(ss) < max_seq_len:  # we add bos token when initialize ss so we need to plus 1
            ss += [vocab.pad_idx] * (max_seq_len - len(ss))
        elif len(ss) > max_seq_len:
            ss = ss[:max_seq_len]
            ss[-1] = vocab.eos_idx
        result.append(ss)
    return np.array(result)

def encode_src_data(data, max_seq_len):
    global src_tokenizer
    result = []
    for i, s in enumerate(data):
        try:
            ss = src_tokenizer.encode(s, add_special_tokens=False,
                                      max_length=src_tokenizer.model_max_length)
            x = ss[:max_seq_len-2]
            x.insert(0, src_tokenizer.bos_token_id)
            x.append(src_tokenizer.eos_token_id)
            gap = max_seq_len - len(x)
            x += [src_tokenizer.pad_token_id] * gap
            result.append(x)
        except:
            print(i)
    return result

src_data_train = load_data_from_file('data/train.en')
trg_data_train, trg_vocab = load_data_from_file('data/train.vi',
                                                build_vocab=True, min_freq=min_freq,
                                                max_vocab_size=max_vocab_size_trg)
train = {'src': src_data_train, 'trg': trg_data_train}
train['src'] = encode_src_data(train['src'], max_seq_len_src)
train['trg'] = encode_trg_data(train['trg'], trg_vocab, max_seq_len_trg)

src_data_val = load_data_from_file('data/tst2012.en')
trg_data_val = load_data_from_file('data/tst2012.vi')
val = {'src': src_data_val, 'trg': trg_data_val}
# val = filter_data_with_lenght(val)
val['src'] = encode_src_data(val['src'], max_seq_len_src)
val['trg'] = encode_trg_data(val['trg'], trg_vocab, max_seq_len_trg)

src_data_test = load_data_from_file('data/tst2013.en')
trg_data_test = load_data_from_file('data/tst2013.vi')
test = {'src': src_data_test, 'trg': trg_data_test}
# val = filter_data_with_lenght(val)
test['src'] = encode_src_data(test['src'], max_seq_len_src)
test['trg'] = encode_trg_data(test['trg'], trg_vocab, max_seq_len_trg)

data = {'trg_vocab': trg_vocab,
        'train': train,
        'valid': val,
        'test': test,
        'max_len': {'src': max_seq_len_src, 'trg': max_seq_len_trg}}

save_data = 'data/m30k_deen_shr.pkl'
print('[Info] Dumping the processed data to pickle file', save_data)
pickle.dump(data, save_data)
