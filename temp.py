import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import html

nlp = en_core_web_sm.load()

s = 'I have a B.A. in English from Harvard College , an MBA in marketing from Wharton Business School .'
doc = nlp(html.unescape(s))
print([(X.text, X.label_) for X in doc.ents])
for X in doc.ents:
    s = s.replace(X.text, X.label_)

from transformers import *

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print(tokenizer.tokenize(s))