""" This functions cleans all the tweets.
It first removes all the #tags, then make sure the tweets
does not contain http links, non ASCII charaters or that the
first letter of the tweet is @ (to ensure that the tweet is not out of context).
Then it removes any @tagging and any mention of the word sarcasm or sarcastic.
If after this the tweet is not empty and contains at least 3 words, it is added to the list.
Finally, duplicate tweets are removed. """

import csv
import re

import nltk
import numpy as np

import enchant

def preprocessing(csv_file_object):

    data = []
    length1 = []
    remove_hashtags = re.compile(r'#\w+\s?')
    remove_friendtag = re.compile(r'@\w+\s?')
    remove_sarcasm = re.compile(re.escape('sarcasm'), re.IGNORECASE)
    remove_sarcastic = re.compile(re.escape('sarcastic'), re.IGNORECASE)
    d = enchant.Dict("en_US")
    toke = nltk.tokenize.RegexpTokenizer(r'\w+')
    for row in csv_file_object:
        if len(row[0:]) == 1:
            temp = row[0:][0]
            temp = remove_hashtags.sub('', temp)
            try:
                temp = temp.decode('string_escape')
            except ValueError:
                print(temp)
                print("Oops!  There's '\\' occur in the last word")
                temp = temp[0:-2]
                print(temp)
                temp = temp.decode('string_escape')
            if (len(temp) > 0) and ('http' not in temp) and (
                    '@' not in temp) and ('\u' not in temp):
                temp = temp.encode('utf8')
                temp = remove_friendtag.sub('', temp)
                temp = remove_sarcasm.sub('', temp)
                temp = remove_sarcastic.sub('', temp)
                temp = ' '.join(temp.split())  # remove useless space
                tokenized = toke.tokenize(temp)
                if len(tokenized) > 2:
                    orilen = len(tokenized)
                    length = 0
                    for word in tokenized:
                        if d.check(word):
                            length += 1
                    if float(length) / float(orilen) > 0.51:
                        data.append(temp)
                        length1.append(orilen)
    data = list(set(data))
    data = np.array(data)
    return data, length1


### POSITIVE DATA ####
csv_file_object_pos = csv.reader(open('crawled_data/twitDB_sarcasm.csv', 'rU'), delimiter='\n')
pos_data, length_pos = preprocessing(csv_file_object_pos)

### NEGATIVE DATA ####
csv_file_object_neg = csv.reader(open('crawled_data/twitDB_regular.csv', 'rU'), delimiter='\n')
neg_data, length_neg = preprocessing(csv_file_object_neg)

print('Number of  sarcastic tweets :', len(pos_data))
print('Average length of sarcastic tweets :', np.mean(length_pos))
print('Number of  non-sarcastic tweets :', len(neg_data))
print('Average length of non-sarcastic tweets :', np.mean(length_neg))

np.save('sarcasm_data_proc', pos_data)
np.save('nonsarc_data_proc', neg_data)
