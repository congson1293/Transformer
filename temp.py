import html, re
from transformers import RobertaTokenizer

def remove_punc(words):
    result = map(lambda w: re.sub('[,.!;:\"\'?<>{}\[\]()-]', '', w), words)
    result = map(lambda w: re.sub('(\d+,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)', 'number', w), result)
    result = list(filter(lambda w: len(w) > 0, result))
    return result

with open('data/train.en', 'r') as fp:
    en = [line for line in fp]
with open('data/train.vi', 'r') as fp:
    vi = [line for line in fp]

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

new_en, new_vi = [], []
for i, line in enumerate(en):
    try:
        x = html.unescape(line.strip())
        x = remove_punc(x.split())
        _ = tokenizer.encode(x)
        new_en.append(line)
        new_vi.append(vi[i])
    except:
        continue

with open('data/new_train.en', 'w') as fp:
    fp.write(''.join(new_en))
with open('data/new_train.vi', 'w') as fp:
    fp.write(''.join(new_vi))
