import re
from . import utils


class Regex:
    def __init__(self):
        self.rm_except_chars = re.compile(
            '[^\w\s\d…\-–\./_,\(\)$%“”\"\'?!;:@#^&*\+=<>\[\]\{\}²³áÁàÀãÃảẢạẠăĂắẮằẰẳẲặẶẵẴâÂấẤầẦẩẨậẬẫẪđĐéÉèÈẻẺẽẼẹẸêÊếẾềỀễỄểỂệỆíÍìÌỉỈĩĨịỊóÓòÒỏỎõÕọỌôÔốỐồỒổỔỗỖộỘơƠớỚờỜởỞỡỠợỢúÚùÙủỦũŨụỤưƯứỨừỪửỬữỮựỰýÝỳỲỷỶỹỸỵỴ]')
        self.normalize_space = re.compile(' +')
        self.multi_newline_regex = re.compile("\n+")
        self.detect_url = re.compile('(https|http|ftp|ssh)://[^\s\[\]\(\)\{\}]+', re.I)
        self.detect_url2 = re.compile(
            '[^\s\[\]\(\)\{\}]+(\.com|\.net|\.vn|\.org|\.info|\.biz|\.mobi|\.tv|\.ws|\.name|\.us|\.ca|\.uk)', re.I)
        self.detect_num = re.compile(r'(\d+\,\d+\w*)|(\d+\.\d+\w*)|(\w*\d+\w*)')
        self.detect_email = re.compile('[^@|\s]+@[^@|\s]+')
        self.detect_datetime = re.compile('\d+[\-/]\d+[\-/]*\d*')
        self.change_to_space = re.compile('\t')
        self.filter_hard_rules = self.filter_hard_rules()
        self.detect_special_mark = re.compile('[,:;\-\(\)\[\]\{\}\<\>“”\"\']')
        self.detect_special_mark3 = re.compile('[/\$%–@#^&*+=]')
        self.detect_special_mark4 = re.compile('\.\.\.|[?!…\.]')
        self.detect_non_vnese = self.detect_non_vietnamese()

    def run_regex_training(self, data):
        s = self.multi_newline_regex.sub('\n', data)
        s = self.detect_num.sub('1', s)  # replaced number to 1
        s = self.detect_url.sub('2', s)
        s = self.detect_url2.sub('0', s)
        s = self.detect_email.sub('3', s)
        s = self.detect_datetime.sub('4', s)
        s = self.change_to_space.sub(' ', s)
        s = self.rm_except_chars.sub('', s)
        if self.filter_hard_rules:
            s = self.filter_hard_rules.sub('5', s)
        s = self.detect_non_vnese.sub('6', s)
        s = self.detect_special_mark.sub('7', s)
        s = self.detect_special_mark3.sub('9', s)
        s = self.normalize_space.sub(' ', s)
        s = self.detect_special_mark4.sub('.', s)

        return s.strip()

    def replace(self, reobj, mask, s):
        values = []
        new_str = s
        bias = 0
        finditer = reobj.finditer(s)
        for m in finditer:
            x = m.regs[0]
            values.append(s[x[0]:x[1]])
            new_str = new_str[:x[0] - bias] + mask + new_str[x[1] - bias:]
            bias += x[1] - x[0] - 1
        return new_str, values

    def run_regex_predict(self, query):
        s = self.multi_newline_regex.sub('\n', query)
        s, number = self.replace(self.detect_num, '1', s)
        s, url = self.replace(self.detect_url, '2', s)
        s, url2 = self.replace(self.detect_url2, '0', s)
        s, email = self.replace(self.detect_email, '3', s)
        s, datetime = self.replace(self.detect_datetime, '4', s)
        s = self.change_to_space.sub(' ', s)
        s = self.rm_except_chars.sub('', s)
        if self.filter_hard_rules:
            s, hard_rules = self.replace(self.filter_hard_rules, '5', s)
        else:
            hard_rules = []
        s, non_vnese = self.replace(self.detect_non_vnese, '6', s)
        s, mark = self.replace(self.detect_special_mark, '7', s)
        s, mark3 = self.replace(self.detect_special_mark3, '9', s)
        s = self.normalize_space.sub(' ', s)
        s, mark4 = self.replace(self.detect_special_mark4, '.', s)
        return s.strip(), number, url, url2, email, datetime, hard_rules, non_vnese, mark, mark3, mark4

    def restore_info(self, q, number, url, url2, email, datetime, hard_rules, non_vnese, mark, mark3, mark4):
        q = Regex.restore_info_ex(q, mark4, '\.')
        q = Regex.restore_info_ex(q, mark3, '9')
        q = Regex.restore_info_ex(q, mark, '7')
        q = Regex.restore_info_ex(q, non_vnese, '6')
        q = Regex.restore_info_ex(q, hard_rules, '5')
        q = Regex.restore_info_ex(q, datetime, '4')
        q = Regex.restore_info_ex(q, email, '3')
        q = Regex.restore_info_ex(q, url2, '0')
        q = Regex.restore_info_ex(q, url, '2')
        q = Regex.restore_info_ex(q, number, '1')
        return q

    def detect_non_vietnamese(self):
        vowel = ['a', 'e', 'i', 'o', 'u', 'y']
        vowel2 = ['a', 'e', 'i', 'o', 'y']
        vowel3 = ['y']
        double_vowel = [w + w for w in vowel]
        double_vowel = list(set(double_vowel) - set(['uu']))
        double_vowel2 = utils.add_to_list(vowel3, vowel)
        double_vowel2 = list(set(double_vowel2) - set(['yy']))
        consonant = ['b', 'c', 'd', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q',
                     'r', 's', 't', 'v', 'x']
        consonant2 = ['b', 'd', 'g', 'h', 'k', 'l', 'q', 'r', 's', 'v', 'x']
        consotant3 = ['m', 'p']
        consonant4 = ['p', 'q']
        consonant5 = ['b', 'c', 'd', 'g', 'n', 'r']
        special_pattern = ['ch', 'gh', 'kh', 'nh', 'ng', 'ph', 'th', 'tr']
        special_pattern2 = ['ae', 'ea', 'ei', 'ey', 'iy', 'oy', 'ya', 'yi', 'yo', 'yu']
        special_pattern3 = ['gh', 'kh', 'ph', 'th', 'tr']
        special_pattern4 = ['ge', 'gy', 'ka', 'ko', 'ku', 'ry']
        english_chars = ['f', 'j', 'w', 'z']
        double_consonant = utils.add_to_list(consonant, consonant)
        double_consonant = list(set(double_consonant) - set(special_pattern))
        non_vietnamese = double_vowel + double_consonant + utils.add_to_list(vowel, consonant2)
        non_vietnamese += consotant3 + special_pattern2 + utils.add_to_list(vowel, special_pattern3)
        non_vietnamese += utils.add_to_list(vowel, utils.add_to_list(consonant, vowel))
        non_vietnamese += special_pattern4 + utils.add_to_list(consonant4, vowel2) + \
                          utils.add_to_list(consonant, double_vowel2) + utils.add_to_list(consonant5, vowel3)
        non_vietnamese = Regex.filter_non_vnese(set(non_vietnamese)) + english_chars
        s = '|'.join(non_vietnamese)
        return re.compile(r'\w*(' + s + r')\w*', re.I)

    @staticmethod
    def filter_non_vnese(s):
        two = filter(lambda x: len(x) == 2, s)
        three = list(set(s) - set(two))
        new_three = []
        for x1 in three:
            flag = False
            if len(x1) != 3: continue
            for x2 in two:
                if x2 in x1:
                    flag = True
                    break
            if not flag: new_three.append(x1)
        return list(two) + new_three

    @staticmethod
    def restore_info_ex(q, data, mask):
        q = q.replace('%', '%%')
        q = re.sub(mask, '%s', q)
        data = tuple(data)
        try:
            q = q % data  # use format string to get best performance
        except:
            pass
        q = q.replace('%%', '%')
        return q

    def filter_hard_rules(self):
        rules = utils.load_hard_rules()
        if rules:
            rgx = r'%s' % '|'.join(rules)
            return re.compile(rgx)
        return None


if __name__ == '__main__':
    r = Regex()
    s = 'Mr.Siro'
    ss = r.detect_non_vnese.sub('nonvnese', s)
