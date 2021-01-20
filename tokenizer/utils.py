# -*- coding: utf-8 -*-
__author__ = 'nobita'

import os
from io import open


# return list of string
def load_data2list_string(data_file):
    data_list = set(); max_length = 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for data in f:
            data = data[:len(data)-1].strip().lower()
            length = data.count(u' ')
            if length > max_length: max_length = length
            data_list.update([data])
    return data_list, max_length


def update_dict(d1, d2):
    for k, v in d1.items():
        d2.update({k.upper():v.upper()})


def update_dict_ex(d1, d2):
    for k, v in d1.items():
        temp = {}
        for kk, vv in v.items():
            temp.update({kk.upper():vv})
        d2.update({k.upper():temp})


def mkdir(dir):
    if (os.path.exists(dir) == False):
        os.mkdir(dir)


def push_data_to_stack(stack, file_path, file_name):
    sub_folder = os.listdir(file_path)
    for element in sub_folder:
        element = file_name + '/' + element
        stack.append(element)


def update_dict_from_value(d1, d2):
    for k, v in d1.items():
        for kk, vv in v.items():
            d2[k].update({vv:kk})
    return


def string2bytearray(s):
    l = [c for c in s]
    return l


def add_to_list(l1, l2):
    l = []
    for x in l1:
        for xx in l2:
            l.append(x+xx)
    return l