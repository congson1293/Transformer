import re
import spacy
import numpy as np
import joblib as pickle

from vocabulary import Vocabulary


src_lang_model = spacy.load('de')
trg_lang_model = spacy.load('en')

share_vocab = True
max_vocab_size_src = 10000
max_vocab_size_trg = 10000
max_seq_len_src = 30
max_seq_len_trg = 30
min_freq = 3


def remove_punc(words):
    result = list(map(lambda w: re.sub('[,.!;:\"\'?<>{}\[\]()]', '', w), words))
    return result


def load_data_from_file(data_file, lang_model, build_vocab=True, min_freq=1, max_vocab_size=5000):
    with open(data_file) as fp:
        data = [lang_model.tokenizer(text.strip()).text for text in fp]
        data = [remove_punc(tok.split()) for tok in data]
        if build_vocab:
            vocab = Vocabulary()
            vocab.build_vocab(data, lower=True, min_freq=min_freq,
                              max_vocab_size=max_vocab_size)
            return data, vocab
        else:
            return data


def filter_data_with_lenght(data):
    global max_seq_len_src, max_seq_len_trg
    result = {'src':[], 'trg': []}
    for i in range(len(data['src'])):
        if len(data['src'][i]) > max_seq_len_src or len(data['trg'][i]) > max_seq_len_trg:
            continue
        result['src'].append(data['src'][i])
        result['trg'].append(data['trg'][i])
    return result

def encode_data(data, vocab, max_seq_len):
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
        if len(ss) < max_seq_len+1: # we add bos token when initialize ss so we need to plus 1
            ss += [vocab.pad_idx] * (max_seq_len - len(ss) + 1)
        elif len(ss) > max_seq_len+1:
            ss = ss[:max_seq_len+1]
            ss[-1] = vocab.eos_idx
        result.append(ss)
    return np.array(result)


src_data_train, src_vocab = load_data_from_file('data/train.de', src_lang_model,
                                                build_vocab=True, min_freq=min_freq,
                                                max_vocab_size=max_vocab_size_src)
print('[Info] Get source language vocabulary size:', len(src_vocab.stoi))
trg_data_train, trg_vocab = load_data_from_file('data/train.en', trg_lang_model,
                                                build_vocab=True, min_freq=min_freq,
                                                max_vocab_size=max_vocab_size_trg)
print('[Info] Get target language vocabulary size:', len(trg_vocab.stoi))

train = {'src': src_data_train, 'trg': trg_data_train}
# train = filter_data_with_lenght(train)
train['src'] = encode_data(train['src'], src_vocab, max_seq_len_src)
train['trg'] = encode_data(train['trg'], trg_vocab, max_seq_len_trg)

src_data_val = load_data_from_file('data/val.de', src_lang_model, build_vocab=False)
trg_data_val = load_data_from_file('data/val.en', trg_lang_model, build_vocab=False)
val = {'src': src_data_val, 'trg': trg_data_val}
# val = filter_data_with_lenght(val)
val['src'] = encode_data(val['src'], src_vocab, max_seq_len_src)
val['trg'] = encode_data(val['trg'], trg_vocab, max_seq_len_trg)

src_data_test = load_data_from_file('data/test2016.de', src_lang_model, build_vocab=False)
trg_data_test = load_data_from_file('data/test2016.en', trg_lang_model, build_vocab=False)
test = {'src': src_data_test, 'trg': trg_data_test}
# test = filter_data_with_lenght(test)
test['src'] = encode_data(test['src'], src_vocab, max_seq_len_src)
test['trg'] = encode_data(test['trg'], trg_vocab, max_seq_len_trg)

if share_vocab:
    print('[Info] Merging two vocabulary ...')
    for w, _ in src_vocab.stoi.items():
        try:
            _ = trg_vocab.stoi[w]
            continue
        except:
            idx = len(trg_vocab.stoi)
            trg_vocab.stoi[w] = idx
            trg_vocab.itos[idx] = w
    trg_vocab.vocab_size = len(trg_vocab.stoi)
    src_vocab = trg_vocab
    print('[Info] Get merged vocabulary size:', len(src_vocab.stoi))

data = {'vocab': {'src': src_vocab, 'trg': trg_vocab},
        'train': train,
        'valid': val,
        'test': test,
        'max_len': {'src': max_seq_len_src, 'trg': max_seq_len_trg}}

save_data = 'data/m30k_deen_shr.pkl'
print('[Info] Dumping the processed data to pickle file', save_data)
pickle.dump(data, save_data)
