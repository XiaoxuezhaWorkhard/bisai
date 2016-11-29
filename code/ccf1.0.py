# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

train = pd.read_csv('../data/train.csv', names = ['CONS_NO', 'LABEL'])
test = pd.read_csv('../data/test.csv', names = ['CONS_NO', 'LABEL'])

reader = pd.read_csv('../data/ALL_USER_YONGDIAN_DATA.zip', compression='zip', iterator=True)
loop = True
chunkSize = 8000000
chunks = []
num = 1

while loop:
    if num == 1:
        header = True
    else:
        header = False
    try:
        chunk = reader.get_chunk(chunkSize)
        df_train = pd.merge(chunk, train)
        df_train.to_csv('../data/df_train.csv', index = False, mode='a', header = header)
        df_test = pd.merge(chunk, test)
        df_test.to_csv('../data/df_test.csv', index = False, mode = 'a', header = header)
        num = num + 1
        del chunk, df_train, df_test
    except StopIteration:
        loop = False
        print "Iteration is stopped."

