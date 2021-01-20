import os.path
import re

from . import utils
from .feature.feature import Feature
from .regex import Regex

root_dir = os.path.dirname(os.path.realpath(__file__))

class SentenceSpliter():
    def __init__(self):
        self.feature_model = None
        self.regex_rule = Regex()
        self.classifier = utils.load(os.path.join(root_dir + '/model', 'model.pkl'))
        if self.classifier is None:
            print("Unable to load model!")
            exit(-1)

    def make_feature(self, file=None):
        self.feature_model = Feature()
        if file is None:
            return [], []
        else:
            features_list, label_list = self.feature_model.gen_feature_matrix(file)
        return features_list, label_list

    def split_paragraph(self, par):
        sens = []
        try:
            paragraph, number, url, url2, email, datetime, hard_rules, non_vnese, mark, mark3, mark4 = \
                self.regex_rule.run_regex_predict(par)
            features, _ = self.make_feature(paragraph)
            if not features:
                sens.append(par)
                return sens
            labels = self.classifier.predict(features)
            idx = 0
            pos_start = 0
            pos_end = 0
            for c in paragraph:
                if Feature.is_splitter_candidate(c):
                    if idx < len(labels) and labels[idx] == 1:
                        sens.append(paragraph[pos_start:pos_end + 1].strip())
                        pos_start = pos_end + 1
                    idx += 1
                pos_end += 1
            if pos_start < len(paragraph):
                sens.append(paragraph[pos_start:].strip())
            paragraph = '\n'.join(sens)
            paragraph = self.regex_rule.restore_info(paragraph, number, url, url2, email, datetime, hard_rules,
                                                     non_vnese, mark, mark3, mark4)
            sens = paragraph.split('\n')
            return sens
        except:
            sens.append(par)
            return sens

    def split(self, pars):
        sens = []
        try:
            pars = pars.replace('\r', '\n')
            pars = re.compile('\n+').sub('\n', pars)
            pars = pars.split('\n')
            for par in pars:
                if par.strip():
                    s = self.split_paragraph(par)
                    sens += s
            return sens
        except:
            sens.append(pars)
            return sens


if __name__ == '__main__':
    spliter = SentenceSpliter()
    print(spliter.split('Theo kết luận điều tra, bị can Nông Văn Lư (36 tuổi, ở Bắc Giang) là lái xe của Công ty TNHH Thương mại và Dịch vụ kỹ thuật Nhật Cường (Công ty Nhật Cường) khai, từ năm 2014 đến tháng 5/2019, theo chỉ đạo của Bùi Quang Huy (Tổng Giám đốc Công ty Nhật Cường) và Trần Ngọc Ánh (Phó Tổng Giám đốc Công ty Nhật Cường), Lư đã nhận hàng lậu từ 6 đường dây vận chuyển (Hùng HP, SRV, Việt LS, Hưng ĐA, SH, Hằng LS). Hàng hóa chủ yếu là điện thoại iphone các loại để trong các thùng carton. Khi Lư nhận, hàng hóa không có hóa đơn, chứng từ.'))