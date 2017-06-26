import enchant
import nltk
import sys
import numpy as np
import re

file = sys.argv[1]
newFile = sys.argv[2]
d = enchant.Dict("en_US")
toke = nltk.tokenize.RegexpTokenizer(r'\w+')
ori_data = np.load(file)
data = [ ]
for datum in ori_data:
    line = datum.strip()
    if "\\" not in datum[-1]:
        line = datum.decode('string-escape')
    tokenized = toke.tokenize(line)
    orilen = len(tokenized)
    length = 0
    for word in tokenized:
        if d.check(word):
            length+=1
    if float(orilen)==0:
        pass
    elif len(line)==0:
        pass
    elif float(length)/float(orilen) > 0.51:
        data.append(line)

np.save(newFile,data)

