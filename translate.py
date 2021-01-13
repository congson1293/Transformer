import argparse
import time
import torch
from Models import Transformer
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import joblib as pickle
import argparse
from Models import init_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import spacy
from vocabulary import Vocabulary


src_lang_model = spacy.load('de')

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def remove_punc(sen):
    result = re.sub('[,.!;:\"\'?<>{}\[\]()]', '', sen)
    return result

def preprocess_input(s):
    global src_lang_model
    sen = src_lang_model.tokenizer(s.strip()).text
    sen = remove_punc(sen)
    return sen

def translate_sentence(sentence, model, opt, src_vocab, trg_vocab):

    sentence = preprocess_input(sentence)
    words = sentence.split()
    indices = [src_vocab.bos_idx]
    for i, w in enumerate(words):
        if i+1 == opt.max_len:
            break
        try:
            idx = src_vocab.stoi[w.lower()]
        except:
            idx = src_vocab.unk_idx
        indices.append(idx)
    indices.append(src_vocab.eos_idx)
    sentence = Variable(torch.LongTensor([indices]))
    if opt.device == 'cuda':
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, src_vocab, trg_vocab, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(text, opt, model, src_vocab, trb_vocab):
    sentences = text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence, model, opt, src_vocab, trb_vocab).capitalize())

    return (' '.join(translated))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-beam_size', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=50)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    opt.load_weights = True

    assert opt.beam_size > 0
    assert opt.max_len > 10

    checkpoint = torch.load('models/checkpoint.chkpt')
    settings = checkpoint['settings']

    vocab = pickle.load('models/vocab.pkl')
    src_vocab = vocab['src']
    trg_vocab = vocab['trg']

    model = Transformer(src_vocab.vocab_size, trg_vocab.vocab_size, settings.d_model,
                        settings.n_layers, settings.heads, settings.dropout)
    
    while True:
        # text = input("Enter a sentence to translate (type 'q' to quit):\n")
        text = 'Leute Reparieren das Dach eines Hauses.'
        if text=="q":
            break
        phrase = translate(text, opt, model, src_vocab, trg_vocab)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
