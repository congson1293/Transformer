from transformers import *

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
x = tokenizer.encode('how are you ?', add_special_tokens=True)
print(x)