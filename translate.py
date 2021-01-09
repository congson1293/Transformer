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
from vocabulary import Vocabulary

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 'cuda':
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(opt, model, SRC, TRG):
    sentences = opt.text.lower().split('.')
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize())

    return (' '.join(translated))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-beam_size', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=50)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()

    opt.device = 'cpu' if opt.no_cuda is False else 'cuda'
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
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, src_vocab, trg_vocab)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
