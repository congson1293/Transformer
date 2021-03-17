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
import en_core_web_sm

src_lang_model = RobertaTokenizer.from_pretrained('roberta-base')
ner = en_core_web_sm.load()
ner_tag = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT',
           'WORK_OF_ART', 'LANGUAGE', 'EVENT', 'LAW']

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
    doc = ner(s)
    sen = s
    entities = {n:[] for n in ner_tag}
    for X in doc.ents:
        if not X.label_ in ner_tag:
            continue
        sen = sen.replace(X.text, X.label_)
        entities[X.label_].append(X.text)
    return sen, entities

def restore_entity(s, entities):
    global ner_tag
    words = s.split()
    result = []
    for w in words:
        try:
            ww = w.upper()
            if ww in ner_tag:
                result.append(entities[ww].pop())
            else:
                result.append(w)
        except:
            result.append(w)

def translate_sentence(sentence, model, opt, src_vocab, trg_vocab):
    global src_lang_model

    s, entities = preprocess_input(sentence)

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

def translate(text, opt, model, src_vocab, trg_vocab):
    sentences = text.lower().split('.')
    sentences = list(filter(lambda s: len(s) > 0, sentences))
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence, model, opt, src_vocab, trg_vocab).capitalize())

    return (' '.join(translated))

def cal_bleu(opt, model, trb_vocab):
    import re, html
    from nltk.translate.bleu_score import sentence_bleu

    def remove_punc(sen):
        result = html.unescape(sen.strip())
        result = re.sub('[,.!;:\"\'?<>{}\[\]()-]', '', result)
        return result.lower()

    global src_lang_model

    bleu = []
    print('calculate bleu score ...')
    with open('data/tst2013.vi', 'r') as fp:
        trg_sentences = [remove_punc(sen) for sen in fp]
    print('there are {} sentences'.format(len(trg_sentences)))
    with open('data/tst2013.en', 'r') as fp:
        for i, sen in enumerate(fp):
            trg_sen = trg_sentences[i]
            pred_sen = remove_punc(translate(sen, opt, model, src_lang_model, trb_vocab))
            bleu.append(sentence_bleu([trg_sen], pred_sen))
            print('\rcalculated bleu score of sentence {}-th ...'.format(i+1), end='', flush=True)
    print('\nCumulative bleu score 4-gram = %.4f' % (sum(bleu)/len(bleu)))


def main():
    global src_lang_model

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

    trg_vocab = pickle.load('models/trg_vocab.pkl')

    model = Transformer(trg_vocab.vocab_size, settings.d_model,
                        settings.n_layers, settings.heads, settings.dropout).to(opt.device)
    model.load_state_dict(checkpoint['model'])

    # cal_bleu(opt, model, trg_vocab)

    while True:
        text = input("Enter a sentence to translate (type 'q' to quit):\n")
        # text = 'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.'
        phrase = translate(text, opt, model, src_lang_model, trg_vocab)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
# Cumulative bleu score 4-gram = 0.5369