import sys
import nltk
import enchant

file = sys.argv[1]
newFile = sys.argv[2]
d = enchant.Dict("en_US")
toke = nltk.tokenize.RegexpTokenizer(r'\w+')
data = []
with open(file, 'r') as oriFile:
    for line in oriFile:
        tokenized = toke.tokenize(line)
        orilen = len(tokenized)
        length = 0
        for word in tokenized:
            if d.check(word):
                length += 1
        if float(length) / float(orilen) > 0.51:
            data.append(line)
with open(newFile, 'w') as newfile:
    for line in data:
        newfile.write(line)
