import torch
from Models import Transformer
import joblib as pickle
import argparse
from Beam import beam_search
from torch.autograd import Variable
import re
import spacy
from vocabulary import Vocabulary
from transformers import RobertaTokenizer


src_lang_model = RobertaTokenizer.from_pretrained('roberta-base')

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def remove_punc(sen):
    result = re.sub('[,.!;:\"\'?<>{}\[\]()]', '', sen)
    result = re.sub('(\d+,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)', 'number', result)
    return result

def preprocess_input(s):
    sen = remove_punc(s)
    return sen

def translate_sentence(sentence, model, opt, src_vocab, trg_vocab):
    global src_lang_model

    s = preprocess_input(sentence)

    indices = src_lang_model.encode(s, add_special_tokens=False)
    indices = indices[:opt.max_src_len - 2]
    indices.insert(0, src_lang_model.bos_token_id)
    indices.append(src_lang_model.eos_token_id)
    gap = opt.max_src_len - len(indices)
    indices += [src_lang_model.pad_token_id] * gap

    sen = Variable(torch.LongTensor([indices]))
    sen = sen.to(opt.device)
    
    sentence = beam_search(sen, model, src_vocab, trg_vocab, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def translate(text, opt, model, src_vocab, trb_vocab):
    sentences = text.lower().split('.')
    sentences = list(filter(lambda s: len(s) > 0, sentences))
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence, model, opt, src_vocab, trb_vocab).capitalize())

    return (' '.join(translated))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-beam_size', type=int, default=3)
    parser.add_argument('-no_cuda', type=bool, default=False)

    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'

    assert opt.beam_size > 0

    checkpoint = torch.load('models/checkpoint.chkpt', map_location=torch.device(opt.device))
    settings = checkpoint['settings']

    opt.max_src_len = settings.max_src_len
    opt.max_trg_len = settings.max_trg_len

    vocab = pickle.load('models/vocab.pkl')
    src_vocab = vocab['src']
    trg_vocab = vocab['trg']

    model = Transformer(src_vocab.vocab_size, trg_vocab.vocab_size, settings.d_model,
                        settings.n_layers, settings.heads, settings.dropout).to(opt.device)
    model.load_state_dict(checkpoint['model'])

    while True:
        text = input("Enter a sentence to translate (type 'q' to quit):\n")
        # text = 'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.'
        phrase = translate(text, opt, model, src_vocab, trg_vocab)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
# Cumulative bleu score 4-gram = 0.5369