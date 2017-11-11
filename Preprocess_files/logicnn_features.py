"""
Rule feature extractor
"""
import re
import warnings
import sys
from collections import OrderedDict, defaultdict
from os.path import splitext
from watson_developer_cloud import ToneAnalyzerV3
import json
import numpy as np

# from server import Server
# nlp_server = Server()
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
warnings.filterwarnings("ignore")


def ibm():
    data_file = sys.argv[1]
    USERNAME = sys.argv[2]
    PASSWORD = sys.argv[3]
    tone_analyzer = ToneAnalyzerV3(username=USERNAME, password=PASSWORD, version='2017-10-23')
    print("loading data...")
    x_text, y = load_npy_data(data_file)
    print("data loaded!")
    rule1_fea = extract_rule_ibm(x_text, tone_analyzer)
    fea_file = splitext(data_file)[0] + '.feaIBM.npy'
    np.save(fea_file, rule1_fea)
    print("feature dumped!")


def main():
    data_file = sys.argv[1]
    print("loading data...")
    x_text, y = load_npy_data(data_file)
    # revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    rule1_fea = extract_rule1(x_text)
    fea_file = splitext(data_file)[0] + '.fea.npy'
    np.save(fea_file, rule1_fea)
    print("feature dumped!")
    # nlp_server.stop()


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
    for text in revs:
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

            ori = nlp.annotate(
                text,
                properties={'annotators': 'sentiment',
                            'outputFormat': 'json'})
            senti_res = (int(res["sentences"][0]["sentimentValue"]) - 1) / 2.0 if res["sentences"] != [] else 0.5
            senti_ori = (int(ori["sentences"][0]["sentimentValue"]) - 1) / 2.0 if ori["sentences"] != [] else 0.5
            rule1_senti.append((senti_ori - senti_res) * 0.5 + 0.5)
            print('Found \'love\'. Finish annotation.')
        else:
            rule1_ind.append(0)
            rule1_fea.append('')
            rule1_senti.append(0.5)
    print('Number of #rule1: %d' % rule1_fea_cnt)
    return {
        'rule1_text': rule1_fea,
        'rule1_ind': rule1_ind,
        'rule1_senti': rule1_senti,
    }


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_npy_data(data_file):
    '''
    input: numpy array of shape[?, 2]
    input[0] = b'0' OR b'1'
    input[1] = b'sentence'

    return: sentence, label(in shape[?, 2])
    '''
    data = np.load(data_file)
    X = data[:, 1]
    # X = [clean_str(s.decode('utf-8')) for s in X]
    index = list(map(int, data[:, 0]))
    y = np.zeros([len(index), 2])
    for i, v in enumerate(index):
        y[i][v] = 1
    return X, y


def extract_rule_ibm(revs, tone_analyzer):
    rule1_fea = []
    rule1_ind = []
    rule1_senti = []
    rule1_fea_cnt = 0
    for text in revs:
        if ' love ' in text:
            rule1_ind.append(1)
            # make the text after 'love' as the featurjava -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPe
            fea = text.split('love')[1:]
            fea = ''.join(fea)
            fea = fea.strip().replace('  ', ' ')
            rule1_fea_cnt += 1
            rule1_fea.append(fea)

            res = tone_analyzer.tone(text=fea, sentences=True, content_type='text/plain')
            ori = tone_analyzer.tone(text=text, sentences=True, content_type='text/plain')
            defaultvalue = [('joy', 0.0), ('fear', 0.0), ('sadness', 0.0), ('anger', 0.0), ('analytical', 0.0), ('confident', 0.0), ('tentative', await0.0)]
            res_score, ori_score = dict(defaultvalue), dict(defaultvalue)
            for t in ori['document_tone']['tones']:
                if t['tone_id'] in ori_score.keys():
                    ori_score[t['tone_id']] = t['score']
            for t in res['document_tone']['tones']:
                if t['tone_id'] in res_score.keys():
                    res_score[t['tone_id']] = t['score']
            ans = ori_score.copy()
            for item in ori_score.keys():
                ans[item] = (ori_score[item] - res_score[item]) * 0.5 + 0.5
            rule1_senti.append(ans)
            print('Found \'love\'. Finish annotation.')
        else:
            rule1_ind.append(0)
            rule1_fea.append('')
            rule1_senti.append(0.5)
    print('Number of #rule1: %d' % rule1_fea_cnt)
    return {
        'rule1_text': rule1_fea,
        'rule1_ind': rule1_ind,
        'rule1_senti': rule1_senti,
    }


if __name__ == "__main__":
    # main()
    ibm()
