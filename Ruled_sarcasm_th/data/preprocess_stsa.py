from __future__ import absolute_import, print_function, division

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from random import randint
np.random.seed(7294258)

def build_data(data_folder, clean_string=True):
    """ Loads data """
    revs = []
    [train_file, dev_file, test_file] = data_folder
    vocab = defaultdict(float) #0-train, 1-dev, 2-test

    def extract_rev_update_dict(_file_source, _i,  _clean_string):
        with open(_file_source, "rb") as f:
            for line in f:
                # example sentance:0 I'm happy
                line = line.strip() # remove space in the beginning and end
                y = int(line[0]) # get label
                rev = []
                rev.append(line[2:].strip())
                if _clean_string:
                    orig_rev = clean_str(" ".join(rev)) # clean corpus by re
                else:
                    #orig_rev = " ".join(rev).lower()
                    orig_rev = " ".join(rev)
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1 # save word to vacabulary container
                datum = {"y":y,
                          "text": orig_rev,
                          "num_words": len(orig_rev.split()),
                          "split": _i} # 0-train, 1-dev, 2-test
                revs.append(datum)

    for i, file_source in enumerate(data_folder):
        extract_rev_update_dict(file_source, i, clean_string)
    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header=f.readline()
        try:
            list_header=header.split()
            vocab_size, layer1_size = list(map(int, list_header))
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        except ValueError as e:
            print("not a number!")
            print(e)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__=="__main__":
    stsa_path = sys.argv[1]
    w2v_file = sys.argv[2]
    train_data_file = "%s/stsa.sarcasm.train" % stsa_path
    dev_data_file = "%s/stsa.sarcasm.dev" % stsa_path
    test_data_file = "%s/stsa.sarcasm.test" % stsa_path
    #train_data_file = "%s/stsa.binary.phrases.train" % stsa_path
    #dev_data_file = "%s/stsa.binary.dev" % stsa_path
    #test_data_file = "%s/stsa.binary.test" % stsa_path
    data_folder = [train_data_file, dev_data_file, test_data_file]
    print("loading data..."),
    revs, vocab = build_data(data_folder, clean_string=False)
    '''
    revs:list of datum
    datum={"y":label,"text":sentence,"num_words":#words in sen,"split":0-train, 1-dev, 2-test}
    vocab:dictionary count # of word appearance
    '''
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("./stsa.sarcasm.p", "wb"))
