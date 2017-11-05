"""
BUT-rule feature extractor

"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
#import theano
#import theano.tensor as T
import re
import warnings
import sys
import time
import json
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
warnings.filterwarnings("ignore")   

def text_after_first(text, part):
    if part in text:
        return ''.join(text.split(part)[1:])
    else:
        return ''

def extract_but(revs):
    but_fea = []
    but_ind = []
    but_fea_cnt = 0
    for rev in revs:
        text = rev["text"]
        if ' but ' in text:
            print(text)
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            print(fea)
            but_fea_cnt += 1
        else:
            but_ind.append(0)
            fea = ''
        but_fea.append(fea)
    print '#but %d' % but_fea_cnt
    return {'but_text': but_fea, 'but_ind': but_ind}

def extract_rule1(revs):
    rule1_fea = []
    rule1_ind = []
    rule1_senti = []
    rule1_fea_cnt = 0
    for rev in revs:
        text = rev["text"]
        if ' love ' in text:
            rule1_ind.append(1)
            # make the text after 'love' as the feature
            fea = text.split('love')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            rule1_fea_cnt += 1
            
            residual = nlp.annotate(
                fea,
                properties={'annotators': 'sentiment',
                            'outputFormat': 'json'})
            ori = nlp.annotate(
                text,
                properties={'annotators': 'sentiment',
                            'outputFormat': 'json'})
            senti_res = (int(residual['sentences'][0]['sentimentValue'])-1)/2.0 if residual['sentences'] != [] else 0.5
            senti_ori = (int(ori['sentences'][0]['sentimentValue'])-1)/2.0 if ori['sentences'] != [] else 0.5
            rule1_senti.append((senti_ori-senti_res)*0.5+0.5)
            print('text: ',text)
            print('fea: ',fea)
            print('sentiment: ',rule1_senti[-1])
        else:
            rule1_ind.append(0)
            fea = ''
            rule1_senti.append(0)
        rule1_fea.append(fea)
    print '#rule1 %d' % rule1_fea_cnt
    return {
        'rule1_text': rule1_fea, 
        'rule1_ind': rule1_ind,
        'rule1_senti': rule1_senti
    }

if __name__=="__main__":
    data_file = sys.argv[1]
    print "loading data..."
    x = cPickle.load(open(data_file,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    #but_fea = extract_but(revs)
    rule1_fea = extract_rule1(revs)
    #cPickle.dump(but_fea, open("%s.fea.p" % data_file, "wb"))
    cPickle.dump(rule1_fea, open("%s.fea.p" % data_file, "wb"))
    print "feature dumped!"

