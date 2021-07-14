import pandas
import csv

LIMIT = 1000
offset = 0
count = 0

all_en = []
all_vi = []

while True:
    try:
        df = pandas.read_csv('data/ted2020.tsv', sep='\t', keep_default_na=False,
                             encoding='utf8', quoting=csv.QUOTE_NONE,
                             skiprows=lambda idx: idx < offset and idx > 0,
                             nrows=LIMIT, header=0)

        if len(df) == 0:
            break

        for i, row in df.iterrows():
            try:
                count += 1
                print(f'\rrow {count}-th', end='', flush=True)
                en = row['en'].strip()
                vi = row['vi'].strip()
                if vi == '' or en == '':
                    continue
                all_en.append(en)
                all_vi.append(vi)
                if len(all_vi) >= 323292:
                    break
            except:
                continue
        offset += LIMIT
    except:
        break

print('')
print(f'there are {len(all_vi)} vietnamese sentences')
print(f'there are {len(all_en)} vietnamese sentences')

with open('data/ted2020.en', 'w') as fp:
    fp.write('\n'.join(all_en))
with open('data/ted2020.vi', 'w') as fp:
    fp.write('\n'.join(all_vi))
