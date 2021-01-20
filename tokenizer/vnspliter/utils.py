import os
from io import open
import joblib


def load(model_path):
    # print('loading %s ...' % (model_path))
    if os.path.isfile(model_path):
        return joblib.load(model_path)
    else:
        return None


def save(model, path):
    # print('saving %s ...' % (path))
    joblib.dump(model, path, compress=True)
    return


def load_hard_rules(path='data/rules.dat'):
    rules = []
    if (os.path.exists(path)):
        f = open(path, encoding='UTF-8')
        rules = f.readlines()
    result = []
    for rule in rules:
        if rule[0] == '#':
            continue
        result.append(rule.strip('\n'))
    return result


def mkdir(dir):
    if (os.path.exists(dir) == False):
        os.mkdir(dir)


def add_to_list(l1, l2):
    l = []
    for x in l1:
        for xx in l2:
            l.append(x + xx)
    return l
