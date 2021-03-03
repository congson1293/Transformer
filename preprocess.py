import re
import spacy
import numpy as np
import joblib as pickle
from transformers import *

from vocabulary import Vocabulary
import en_core_web_sm

import html

src_lang_model = RobertaTokenizer.from_pretrained('roberta-base')
ner = en_core_web_sm.load()

ner_tag = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
           'WORK_OF_ART', 'LANGUAGE', 'EVENT', 'LAW']

max_vocab_size_trg = 25000
max_seq_len_src = 100
max_seq_len_trg = 100
min_freq = 1


def remove_punc(words):
    result = map(lambda w: re.sub('[,.!;:\"\'?<>{}\[\]()-]', '', w), words)
    result = map(lambda w: re.sub('(\d+,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)', 'number', w), result)
    result = list(filter(lambda w: len(w) > 0, result))
    return result

def load_data_from_file(data_file):
    global src_lang_model, trg_lang_model
    print(f'loading data from {data_file} ...')
    with open(data_file) as fp:
        data = []
        for i, text in enumerate(fp):
            data.append(html.unescape(text.strip()))
            print(f'\rprocessed {i+1} sentences ...', end='', flush=True)
        print('')
        return data

def encode_ner(src, trg):
    global ner, ner_tag
    new_src, new_trg = [], []
    for i, s in enumerate(src):
        doc = ner(s)
        t = trg[i]
        for X in doc.ents:
            if not X.label_ in ner_tag:
                continue
            s = s.replace(X.text, X.label_)
            t = t.replace(X.text, X.label_)
        new_src.append(s)
        new_trg.append(t)
        print(f'\rencode ner for sentence {i+1}', end='', flush=True)
    return new_src, new_trg

def encode_trg_data(data, vocab, max_seq_len):
    result = []
    for s in data:
        ss = [vocab.bos_idx]
        for w in s.split():
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
    global src_lang_model
    result = []
    for i, s in enumerate(data):
        try:
            ss = src_lang_model.encode(s, add_special_tokens=False)
            x = ss[:max_seq_len-2]
            x.insert(0, src_lang_model.bos_token_id)
            x.append(src_lang_model.eos_token_id)
            gap = max_seq_len - len(x)
            x += [src_lang_model.pad_token_id] * gap
            result.append(x)
        except:
            print(i)
    return result

train = {}
src_data_train = load_data_from_file('data/train.en')
trg_data_train = load_data_from_file('data/train.vi')
src_data_train, trg_data_train = encode_ner(src_data_train, trg_data_train)
trg_vocab = Vocabulary()
trg_vocab.build_vocab(trg_data_train, lower=True, min_freq=min_freq, max_vocab_size=max_vocab_size_trg)
train['src'] = encode_src_data(src_data_train, max_seq_len_src)
train['trg'] = encode_trg_data(trg_data_train, trg_vocab, max_seq_len_trg)

val = {}
src_data_val = load_data_from_file('data/tst2012.en')
trg_data_val = load_data_from_file('data/tst2012.vi')
src_data_val, trg_data_val = encode_ner(src_data_val, trg_data_val)
val['src'] = encode_src_data(src_data_val, max_seq_len_src)
val['trg'] = encode_trg_data(trg_data_val, trg_vocab, max_seq_len_trg)

test = {}
src_data_test = load_data_from_file('data/tst2013.en')
trg_data_test = load_data_from_file('data/tst2013.vi')
src_data_test, trg_data_test = encode_ner(src_data_test, trg_data_test)
test['src'] = encode_src_data(src_data_test, max_seq_len_src)
test['trg'] = encode_trg_data(trg_data_test, trg_vocab, max_seq_len_trg)

data = {'trg_vocab': trg_vocab,
        'train': train,
        'valid': val,
        'test': test,
        'max_len': {'src': max_seq_len_src, 'trg': max_seq_len_trg}}

save_data = 'data/m30k_deen_shr.pkl'
print('[Info] Dumping the processed data to pickle file', save_data)
pickle.dump(data, save_data)
