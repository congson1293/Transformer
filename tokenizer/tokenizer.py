# -*- coding: utf-8 -*-
import unicodedata

__author__ = 'nobita'

from .regex import regex
from .map import my_map
from . import environment as env
import os
from . import utils
import joblib
import re
from .vnspliter.sentence_spliter import SentenceSpliter
import warnings


warnings.filterwarnings('ignore', category=UserWarning)
root_dir = os.path.dirname(os.path.realpath(__file__))

class Tokenizer:
    def __init__(self):
        self.re = regex()
        self.map = my_map()
        self.clf = None
        self.vocab = None
        self.re = regex()
        self.map = my_map()
        self.strong_learner = None
        self.vocab = None
        self.max_length = None
        self.spliter = SentenceSpliter()
        Tokenizer.run(self)


    def load_vocab(self):
        self.vocab = self.load(os.path.join(root_dir, env.VOCAB))
        self.max_length = self.load(os.path.join(root_dir, env.MAX_LENGTH))


    def pre_processing(self, data):
        return self.re.run_regex_predict(data)


    def load(self, model):
        print('loading %s ...' % (model))
        if os.path.isfile(model):
            return joblib.load(model)
        else: return None


    def is_existed(self, d, s):
        try:
            _ = d[s]
            return True
        except:
            d.update({s:True}) # value use boolean to save memory
            return False


    def build_vector(self, data, i):
        num_syllable = 0
        x = env.WINDOW_LENGTH
        train = [0 for _ in range(env.NUM_DIMENSIONS)]
        for j in range(i-1, i-env.WINDOW_LENGTH, -1):
            x -= 1
            if j < 0: break
            num_syllable = self.get_value(train, data, x, j, num_syllable)
            if num_syllable > env.MAX_SYLLABLE: break
        num_syllable = 0
        x = env.WINDOW_LENGTH
        for j in range(i+1, i+env.NUM_DIMENSIONS-env.WINDOW_LENGTH):
            if j >= len(data): break
            num_syllable = self.get_value(train, data, x, j, num_syllable)
            x += 1
            if num_syllable >= env.MAX_SYLLABLE: break
        return train


    def get_value(self, train, data, x, j, count_space):
        i = count_space
        w = data[j]
        if w == u' ' or w == u'_':
            i += 1
        try:
            train[x] = self.map.char2int[w]
        except: train[x] = 0
        return i


    def get_sentences(self, content):
        sentences = self.spliter.split(content)
        sentences = map(lambda x: x.strip(u'.'), sentences)
        return sentences


    def predict(self, query):
        query = unicodedata.normalize('NFKC', query.strip())
        query = query.rstrip(u'.')
        q = self.longest_matching(query)
        sentences = self.get_sentences(q)
        q = u'\n'.join(sentences)
        q, number, url, url2, email, datetime, non_vnese, all_caps, \
        mark, mark2, mark3 = self.pre_processing(q)
        sentences = list(filter(lambda x: x != u'.' and len(x) > 0,
                                map(lambda xx: xx.strip(), q.split(u'\n'))))
        X = []
        true_label = {}
        map_index = {}
        index = 0
        i = 0
        dora = u''
        for sen in sentences:
            # sentences[k] = self.detect_non_vnese_compound_2(sentences[k])
            sen = sen.strip(u'.').strip()
            for j, c in enumerate(sen):
                if c == u' ':
                    v = self.build_vector(sen, j)
                    if self.is_skip(v):
                        true_label.update({i: 0})
                    elif self.detect_non_vnese_compound(v):
                        true_label.update({i: 1})
                    else:
                        X.append(v)
                        map_index.update({index: i})
                        index += 1
                i += 1
            i += 2  # plus 2 for add '.\n' to join sentence
        if len(X) > 0:
            label_predict = self.clf.predict(X)
            dora += self.get_result(u'.\n'.join(sentences), label_predict, true_label, map_index)
            dora = self.restore_info(dora, number, url, url2, email, datetime, non_vnese, all_caps, mark, mark2, mark3)
            dora = self.clean_failed_underscore(dora)
            dora += u'.'
        else:
            dora += self.restore_info(q, number, url, url2, email, datetime, non_vnese, all_caps, mark, mark2, mark3)
            dora += u'.'
        return dora.strip(u'_')


    def clean_failed_underscore(self, data):
        words = map(lambda w: w.strip(u'_'), data.split(u' '))
        return u' '.join(words)


    def get_result(self, data, label_predict, true_label, map_index):
        s = utils.string2bytearray(data)
        for i, l in true_label.items():
            c = self.get_char(l)
            s[i] = c
        for i in map_index.keys():
            c = self.get_char(label_predict[i])
            s[map_index[i]] = c
        return u''.join(s)


    def get_char(self, label):
        if label == 1: return u'_'
        else: return u' '


    def restore_info(self, q, number, url, url2, email, datetime, non_vnese, all_caps, mark, mark2, mark3):
        q = self.restore_info_ex(q, mark3, u'9')
        q = self.restore_info_ex(q, mark2, u'8')
        q = self.restore_info_ex(q, mark, u'7')
        q = self.restore_info_ex(q, non_vnese, u'5')
        q = self.restore_info_ex(q, all_caps, u'6')
        q = self.restore_info_ex(q, datetime, u'4')
        q = self.restore_info_ex(q, email, u'3')
        q = self.restore_info_ex(q, url2, u'0')
        q = self.restore_info_ex(q, url, u'2')
        q = self.restore_info_ex(q, number, u'1')
        return q


    def restore_info_ex(self, q, data, mask):
        q = q.replace(u'%', u'%%')
        q = re.sub(mask, u'%s', q)
        data = tuple(data)
        try:
            q = q % data # use format string to get best performance
        except: pass
        q = q.replace(u'%%', u'%')
        return q


    '''
    function dectect group of non vietnamese and treat them as compound word
    '''
    def detect_non_vnese_compound(self, v):
        i = int(env.NUM_DIMENSIONS/2)
        if v[i] == 183 and v[i-1] == 183:
            return True
        else: return False


    def detect_non_vnese_compound_2(self, sen):
        words = sen.split(u' ')
        if len(words) < 3: return sen
        res = []; i = 0
        while i < len(words):
            try:
                if words[i].istitle() and words[i+1] == u'5' and words[i+2].istitle():
                    res.append(u'_'.join(words[i:i+3]))
                    i += 3
                elif words[i] == u'5' and words[i-1].istitle() and words[i+1].istitle():
                    _ = res.pop()
                    res.append(u'_'.join(words[i-1:i+2]))
                    i += 2
                else:
                    res.append(words[i]); i += 1
            except: res.append(words[i]); i += 1
        return u' '.join(res)


    def is_skip(self, v):
        i = int(env.NUM_DIMENSIONS/2)
        if self.is_skip_all_caps_non_vnese(v):
            return True
        try:
            _ = self.map.special_characters[v[i]]
            return True
        except:
            try:
                _ = self.map.special_characters[v[i-1]]
                return True
            except: return False

    def is_skip_all_caps_non_vnese(self, v):
        i = int(env.NUM_DIMENSIONS / 2)
        if v[i] == 184 and v[i - 1] == 184:
            return True
        return False


    def longest_matching(self, q):
        try:
            _ = q.index(u' ')
        except: return q
        ambiguous_info = {}; ambiguous = []
        sentences = self.get_sentences(q)
        for k, sentence in enumerate(sentences):
            words = sentence.strip().split(u' ')
            i = 0; sen = []
            while i < len(words):
                w = self.re.normalize_special_mark.sub(u'', words[i])
                s = words[i]
                if len(words) == 1:
                    sen.append(words[0])
                    break
                for l in range(min(self.max_length, len(words)) - 1, 0, -1):
                    try:
                        d = self.vocab[l][w.lower()]
                        ss = u' '.join(words[i:i+l+1])
                        if self.re.normalize_special_mark.search(ss) != None:
                            sss = self.re.normalize_special_mark.sub(u'', ss)
                        else: sss = ss
                        _ = d[sss.lower()]
                        if l > 1:
                            sen.append(ss.replace(u' ', u'_'))
                            i += l+1; break
                        else: ll, sss = self.verify_longest_matching(words, i+1)
                        if ll > l:
                            sen.extend([words[i], sss.replace(u' ', u'_')])
                            i += l + ll + 1
                        elif ll == l:
                            ambiguous.append(tuple([len(sen), k]))
                            j = i + 2
                            sen.extend(words[i:j]); i = j
                        else: sen.append(ss.replace(u' ', u'_')); i += l+1
                        break
                    except:
                        if l == 1: sen.append(s); i += 1
                        continue
            new_sentence = u' '.join(sen)
            ambiguous_info.update({k:[new_sentence, sen]})
        result = self.process_ambiguous(ambiguous_info, ambiguous)
        return result


    def verify_longest_matching(self, words, i):
        w = words[i]; s = u''
        for l in range(self.max_length, 0, -1):
            try:
                d = self.vocab[l][w.lower()]
                ss = u' '.join(words[i:i+l+1])
                _ = d[ss.lower()]
                s = ss.replace(u' ', u'_')
                break
            except: continue
        return s.count(u'_'), s


    def process_ambiguous(self, ambiguous_info, ambiguous):
        X = []
        offset = [0 for _ in ambiguous_info.keys()]
        for k in ambiguous:
            sentence = ambiguous_info[k[1]][0]; words = ambiguous_info[k[1]][1]
            self.process_ambiguous_ex(sentence, words, k[0], X)
        if len(X) > 0:
            prob = self.clf.predict_proba(X)
        else: # out of ambiguous
            res = [ambiguous_info[k][0] for k in range(len(ambiguous_info))]
            return u'\n'.join(res)
        for i, k in enumerate(ambiguous):
            words = ambiguous_info[k[1]][1]
            p1 = prob[i*2][1]; p2 = prob[i*2+1][1]
            index = k[0]-offset[k[1]]
            if p1 > p2:
                words[index] = u'_'.join(words[index:index+2])
                del words[index+1]
            else:
                words[index+1] = u'_'.join(words[index+1:index+3])
                del words[index+2]
            offset[k[1]] += 1
        for k, v in ambiguous_info.items():
            v[0] = u' '.join(v[1])
        s = [x[0] for x in ambiguous_info.values()]
        return u'\n'.join(s)


    def process_ambiguous_ex(self, sentence, words, i, X):
        x = len(u' '.join(words[:i+1]))
        v1 = self.build_vector(sentence, x)
        X.append(v1)
        xx = len(u' '.join(words[:i+2]))
        v2 = self.build_vector(sentence, xx)
        X.append(v2)


    @staticmethod
    def run(self):
        self.load_vocab()
        self.clf = self.load(os.path.join(root_dir, env.MODEL))



if __name__ == '__main__':
    t = Tokenizer()
    s = u'tiền vệ'
    print(t.predict(s))