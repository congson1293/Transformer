# -*- encoding: utf-8 -*-
__author__ = 'nobita'

import re
from . import utils


class regex:
    def __init__(self):
        self.rm_except_chars = re.compile(u'[^\w\s\d…\-–\./_,\(\)$%“”\"\'?!;:@#^&*\+=<>\[\]\{\}²³áÁàÀãÃảẢạẠăĂắẮằẰẳẲặẶẵẴâÂấẤầẦẩẨậẬẫẪđĐéÉèÈẻẺẽẼẹẸêÊếẾềỀễỄểỂệỆíÍìÌỉỈĩĨịỊóÓòÒỏỎõÕọỌôÔốỐồỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢúÚùÙủỦũŨụỤưƯứỨừỪửỬữỮựỰýÝỳỲỷỶỹỸỵỴ]')
        self.normalize_space = re.compile(u' +')
        self.detect_url = re.compile(u'(https|http|ftp|ssh)://[^\s\[\]\(\)\{\}]+', re.I)
        self.detect_url2 = re.compile(u'[^\s\[\]\(\)\{\}]+(\.com|\.net|\.vn|\.org|\.info|\.biz|\.mobi|\.tv|\.ws|\.name|\.us|\.ca|\.uk)', re.I)
        self.detect_num = re.compile(r'(\d+\,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)')
        self.detect_email = re.compile(u'[^@|\s]+@[^@|\s]+')
        self.detect_datetime = re.compile(u'\d+[\-/]\d+[\-/]*\d*')
        self.change_to_space = re.compile(u'\t')
        self.normalize_special_mark = re.compile(u'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\'/\<\>])')
        self.detect_special_mark = re.compile(u'[\(\)\[\]\{\}\<\>“”\"\']')
        self.detect_special_mark2 = re.compile(u'[\.;!?:,]')
        # self.detect_special_mark3 = re.compile(u'[/\-$%–@#^&*+=]')
        self.detect_special_mark3 = re.compile(u'[/\$%–@#^&*+=]')
        # detect non-vietnamese words
        self.detect_non_vnese = self.detect_non_vietnamese()


    def run_regex_training(self, data):
        s = self.detect_num.sub(u'1', data) # replaced number to 1
        s = self.detect_url.sub(u'2', s)
        s = self.detect_url2.sub(u'0', s)
        s = self.detect_email.sub(u'3', s)
        s = self.detect_datetime.sub(u'4', s)
        s = self.change_to_space.sub(u' ', s)
        s = self.rm_except_chars.sub(u'', s)
        s = self.detect_all_caps(u'6', s)
        s = self.detect_non_vnese.sub(u'5', s)
        s = self.normalize_special_mark.sub(u' \g<special_mark> ', s)
        s = self.detect_special_mark.sub(u'7', s)
        s = self.detect_special_mark2.sub(u'8', s)
        s = self.detect_special_mark3.sub(u'9', s)
        s = self.normalize_space.sub(u' ', s)
        return s.strip()


    def replace(self, reobj, mask, s):
        values = []; new_str = s; bias = 0
        finditer = reobj.finditer(s)
        for m in finditer:
            x = m.regs[0]
            values.append(s[x[0]:x[1]])
            new_str = new_str[:x[0]-bias] + mask + new_str[x[1]-bias:]
            bias += x[1]-x[0] - 1
        return new_str, values


    def run_regex_predict(self, query):
        s, number = self.replace(self.detect_num, u'1', query)
        s, url = self.replace(self.detect_url, u'2', s)
        s, url2 = self.replace(self.detect_url2, u'0', s)
        s, email = self.replace(self.detect_email, u'3', s)
        s, datetime = self.replace(self.detect_datetime, u'4', s)
        s = self.change_to_space.sub(u' ', s)
        s = self.rm_except_chars.sub(u'', s)
        s, all_caps = self.detect_all_caps(u'6', s, predict_mode=True)
        s, non_vnese = self.replace(self.detect_non_vnese, u'5', s)
        s = self.normalize_special_mark.sub(u' \g<special_mark> ', s)
        s, mark = self.replace(self.detect_special_mark, u'7', s)
        s, mark2 = self.replace(self.detect_special_mark2, u'8', s)
        s, mark3 = self.replace(self.detect_special_mark3, u'9', s)
        s = self.normalize_space.sub(u' ', s)
        return s.strip(), number, url, url2, email, datetime, \
               non_vnese, all_caps, mark, mark2, mark3


    def detect_non_vietnamese(self):
        vowel = [u'a', u'e', u'i', u'o', u'u', u'y']
        vowel2 = [u'a', u'e', u'i', u'o', u'y']
        vowel3 = [u'y']
        double_vowel = [w+w for w in vowel]
        double_vowel = list(set(double_vowel)-set([u'uu']))
        double_vowel2 = utils.add_to_list(vowel3, vowel)
        double_vowel2 = list(set(double_vowel2)-set([u'yy']))
        consonant = [u'b', u'c', u'd', u'g', u'h', u'k', u'l', u'm', u'n', u'p', u'q',
                     u'r', u's', u't', u'v', u'x']
        consonant2 = [u'b', u'd', u'g',  u'h',  u'k', u'l', u'q', u'r',  u's', u'v', u'x']
        consotant3 = [u'm',  u'p']
        consonant4 = [u'p', u'q']
        consonant5 = [u'b', u'c', u'd', u'g', u'n', u'r']
        special_pattern = [u'ch', u'gh', u'kh', u'nh', u'ng', u'ph', u'th', u'tr']
        special_pattern2 = [u'ae', u'ea', u'ei', u'ey', u'iy', u'oy', u'ya', u'yi', u'yo', u'yu']
        special_pattern3 = [u'gh', u'kh', u'ph', u'th', u'tr']
        special_pattern4 = [u'ge', u'gy', u'ka', u'ko', u'ku', u'ry']
        english_chars = [u'f', u'j', u'w', u'z']
        double_consonant = utils.add_to_list(consonant, consonant)
        double_consonant = list(set(double_consonant) - set(special_pattern))
        non_vietnamese = double_vowel + double_consonant + utils.add_to_list(vowel, consonant2)
        non_vietnamese += consotant3 + special_pattern2 + utils.add_to_list(vowel, special_pattern3)
        non_vietnamese += utils.add_to_list(vowel, utils.add_to_list(consonant, vowel))
        non_vietnamese += special_pattern4 + utils.add_to_list(consonant4, vowel2) + \
                          utils.add_to_list(consonant, double_vowel2) + utils.add_to_list(consonant5, vowel3)
        non_vietnamese = self.filter_non_vnese(set(non_vietnamese)) + english_chars
        s = u'|'.join(non_vietnamese)
        preposition_patterns = r'( in | on | at | a |the |that | it )'
        aggregate_patterns = r'(\w*(' + s + r')\w*)' + u'|' + preposition_patterns
        # return re.compile(ur'\w*(' + s + ur')\w*', re.I)
        return re.compile(aggregate_patterns, re.I)


    def filter_non_vnese(self, s):
        two = list(filter(lambda x: len(x) == 2, s))
        three = list(set(s) - set(two))
        new_three = []
        for x1 in three:
            flag = False
            if len(x1) != 3: continue
            for x2 in two:
                if x2 in x1:
                    flag = True; break
            if not flag: new_three.append(x1)
        return two + new_three


    def detect_all_caps(self, mask, s, predict_mode=False):
        original = s.split(u' ')
        new = []; values = []
        for w in original:
            if w.isupper():
                new.append(mask)
                if predict_mode:
                    values.append(w)
            else: new.append(w)
        if not predict_mode:
            return u' '.join(new)
        else: return u' '.join(new), values



if __name__ == '__main__':
    r = regex()
    s = u'ngày 25.12.2018'
    ss = r.detect_datetime.sub(u'datetime', s)
    print(ss)