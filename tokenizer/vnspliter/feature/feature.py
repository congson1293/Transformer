from .map import CharMap


class Feature:
    NONE_SPLITER_OFFSET = 100
    NEXT_LOCAL_STEP = 15
    MAX_NAME_LENGTH = 15
    HALF_OFFSET_VECTOR = 7
    CHAR_MAP = CharMap()
    SPLITTER_CHAR = 200

    def gen_feature_vector(self, str=None, pos_start=0, pos_end=0):

        features = []
        for i in range(pos_start, pos_end):
            features.append(Feature.char2int(str, i))
        return features

    def gen_feature_matrix(self, str):
        features_list = []
        label_list = []
        idx = 0
        str_len = len(str)
        while idx < str_len:
            current_char = str[idx]
            try:
                next_char = str[idx + 1]
            except:
                next_char = '\n'
            pos_start = idx - Feature.HALF_OFFSET_VECTOR
            pos_end = idx + Feature.HALF_OFFSET_VECTOR + 1
            if Feature.is_splitter_candidate(current_char):
                if Feature.is_new_line_char(next_char):
                    features_list.append(self.gen_feature_vector(str, pos_start, pos_end))
                    label_list.append(1)
                else:
                    features_list.append(self.gen_feature_vector(str, pos_start, pos_end))
                    label_list.append(0)
            idx += 1
        return features_list, label_list

    @staticmethod
    def char2int(str, idx=0):
        if idx <= 0 or idx >= len(str):
            return 0
        elif Feature.is_new_line_char(str[idx]):
            return Feature.CHAR_MAP.char2int[u' ']
        else:
            try:
                return Feature.CHAR_MAP.char2int[str[idx]]
            except:
                return Feature.CHAR_MAP.except_value

    @staticmethod
    def is_splitter_candidate(char):
        return char == u"." or char == u'!' or char == u'?'

    @staticmethod
    def is_new_line_char(char):
        return char == "\n" or char == "\r"
