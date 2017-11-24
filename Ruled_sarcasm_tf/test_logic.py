#! /usr/bin/env python3

import datetime
import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import accuracy_score

try:
    import cPicle as pickle
except:
    import pickle

tf.flags.DEFINE_string("data_file", "../Data/test_balanced.npy", "Data source for the testing data.")
# tf.flags.DEFINE_string("fea_file", "../Data/train_balanced.fea.npy", "Data source for the training feature data.")
tf.flags.DEFINE_boolean("train_word2vec", False, "Whether training use train word2vec (default: False)")
tf.flags.DEFINE_string("model_ckpt", " ", "path to checkpoit of trained model")
tf.flags.DEFINE_float("gpu_usage", 0.2, "per process gpu memory fraction, (defult: 1.0)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Load data
print("Loading data...")
x_text, y = data_helpers.load_npy_data(FLAGS.data_file)
max_document_length = 46

if not FLAGS.train_word2vec:
    print("Direct use word embeddings ...")
    print("Loading word embeddings ...")
    w2v_path = '../Data/GoogleNews-vectors-negative300.bin'
    w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # Build vocabulary
    vocabs = set()
    print("Transform raw text to word_vectors")
    UNK_embed = np.zeros(w2v.vector_size)
    vocabs.add('<UNK>')
    x_w2v = [s.split() for s in x_text]
    for i, s in enumerate(x_w2v):
        for j, w in enumerate(s):
            try:
                s[j] = w2v[w]
                vocabs.add(w)
            except KeyError as err:
                s[j] = UNK_embed

        if len(s) < max_document_length:
            s = [UNK_embed] * (max_document_length - len(s)) + s
            # s = s + [UNK_embed] * (max_document_length - len(s))
        x_w2v[i] = np.asarray(s)
    x_w2v = np.asarray(x_w2v)

else:
    vocab_processor = pickle.load(open('../Data/vocab_processor', 'rb'))
    vocab_processor.vocabulary_.freeze(True)
    x_w2v = np.array(list(vocab_processor.transform(x_text)))

print("Loading trained model from {}".format(FLAGS.model_ckpt))
tf.reset_default_graph()
session_conf = tf.ConfigProto(
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage),
    device_count={"GPU": 0, "CPU": 4},
    allow_soft_placement=True,
    log_device_placement=False)

with tf.Session(config=session_conf) as sess:
    saver = tf.train.import_meta_graph(FLAGS.model_ckpt + '.meta', clear_devices=True)
    saver.restore(sess, FLAGS.model_ckpt)

    Graph = tf.get_default_graph()
    input_x = Graph.get_tensor_by_name('input_x:0')
    input_y = Graph.get_tensor_by_name('input_y:0')
    dropout = Graph.get_tensor_by_name('dropout_keep_prob:0')

    pred_T = Graph.get_tensor_by_name('output/predictions:0')
    # loss_T = Graph.get_tensor_by_name('loss/loss:0')
    acc_T = Graph.get_tensor_by_name('accuracy/accuracy:0')

    feed_dict = {
        input_x: x_w2v,
        input_y: y,
        dropout: 1.,
        # logic_nn.pi: pi,
        # rule1_ind: np.expand_dims(x_fea_batch["rule1_ind"], 1),
        # rule1_senti: np.expand_dims(x_fea_batch["rule1_senti"], 1),
    }
    
    love_collection = []
    for s in x_text:
        if ' love ' in s:
            love_collection.append(1)
        else:
            love_collection.append(0) 
    print("Testing ...")
    pred, acc = sess.run([pred_T, acc_T], feed_dict)
    print(pred)
    print("accuracy = {}".format(acc))
    
    love = []
    non_love = []
    with open('_correct_predict', 'w') as f:
        for idx, label_love in enumerate(love_collection):
            if label_love == 1:
                if int(y[idx][1])==pred[idx]:
                    love.append(1)
                    f.write(str(idx)+'\n')
                else:
                    love.append(0)
            elif label_love == 0:
                if int(y[idx][1])==pred[idx]:
                    non_love.append(1)
                else:
                    non_love.append(0)
    print('Love accuracy: ', accuracy_score(np.asarray(love), np.ones((len(love),))))
    print('Love data: ', len(love))
    print('Others accuracy: ', accuracy_score(np.asarray(non_love), np.ones((len(non_love),))))
    print('Others data: ', len(non_love))
