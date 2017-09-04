"""
Rule feature extractor
"""
import cPickle
import re
import sys
import warnings
from collections import OrderedDict, defaultdict

import numpy as np

# from server import Server
# nlp_server = Server()
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

warnings.filterwarnings("ignore")


def text_after_first(text, part):
    if part in text:
        return ''.join(text.split(part)[1:])
    else:
        return ''


"""
def extract_but(revs):
    but_fea = []
    but_ind = []
    but_fea_cnt = 0
    for rev in revs:
        text = rev["text"]
        if ' but ' in text:
            but_ind.append(1)
            # make the text after 'but' as the feature
            fea = text.split('but')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            but_ind.append(0)
            fea = ''
        but_fea.append(fea)
    print '#but %d' % but_fea_cnt
    return {'but_text': but_fea, 'but_ind': but_ind}
"""


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
            rule1_fea.append(fea)

            res = nlp.annotate(
                fea,
                properties={'annotators': 'sentiment',
                            'outputFormat': 'json'})
            senti = (int(res["sentences"][0]["sentimentValue"]) - 1) / 2.0 if res["sentences"] != [] else 0.5
            rule1_senti.append(senti)
        else:
            rule1_ind.append(0)
            rule1_fea.append('')
            rule1_senti.append(0.5)
    print '#rule1 %d' % rule1_fea_cnt
    return {
        'rule1_text': rule1_fea,
        'rule1_ind': rule1_ind,
        'rule1_senti': rule1_senti
    }


if __name__ == "__main__":
    data_file = sys.argv[1]
    print "loading data..."
    x = cPickle.load(open(data_file, "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    rule1_fea = extract_rule1(revs)
    cPickle.dump(rule1_fea, open("%s.fea.p" % data_file, "wb"))
    print "feature dumped!"
    # nlp_server.stop()
