#! /usr/bin/env python3

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers
from fol import FOL_Rule1
from gensim.models.keyedvectors import KeyedVectors
from logic_nn import LogicNN
from text_cnn import TextCNN

try:
    import cPicle as pickle
except:
    import pickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../Data/sarcasm_data_proc.npy", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../Data/nonsarc_data_proc.npy", "Data source for the negative data.")
tf.flags.DEFINE_string("data_file", "../Data/train_balanced.npy", "Data source for the training data.")
tf.flags.DEFINE_string("fea_file", "../Data/train_balanced.fea.npy", "Data source for the training feature data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_boolean("word2vec", False, "Using pre-trained word2vec as initializer (default: False)")
tf.flags.DEFINE_boolean("train_word2vec", False, "Whether to train word2vec (default: False)")
tf.flags.DEFINE_string("pi_params", "0.95,0", "parameters of pi: 'base of decay func, lower bound' (default: '0.95,0 ')")
tf.flags.DEFINE_string("pi_curve", "exp_arise", "type of pi change curve: exp_arise, exp_decay,linear_arise, linear_decay (default: exp_arise)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs (default: 80)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_usage", 1.0, "per process gpu memory fraction, (defult: 1.0)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
FLAGS.pi_params = list(map(float, FLAGS.pi_params.split(",")))

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_npy_data(FLAGS.data_file)
x_fea = np.load(FLAGS.fea_file).item()
max_document_length = max([len(x.split(' ')) for x in x_text])

if FLAGS.train_word2vec:
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_w2v = np.array(list(vocab_processor.fit_transform(x_text)))
    pickle.dump(vocab_processor, open('../Data/vocab_processor', 'wb'))
else:
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

# Split train/dev set
# TODO: This is very crude, should use cross-validation
print("Split train and dev sets")
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_w2v[:dev_sample_index], x_w2v[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
x_fea_train = {}
x_fea_dev = {}
for k, v in x_fea.items():
    x_fea_train[k] = v[:dev_sample_index]
    x_fea_dev[k] = v[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_) if FLAGS.train_word2vec else len(vocabs) ))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_usage),
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            vocab_size=len(vocab_processor.vocabulary_) if FLAGS.train_word2vec else len(vocabs),
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            train_w2v=FLAGS.train_word2vec
        )

        # build the feature of RULE1-rule
        # fea: symbolic feature tensor, of shape 3
        #      fea[0]   : 1 if x=x1_love_x2, 0 otherwise
        #      fea[1:2] : classifier.predict_p(x_2)

        rule1_ind = tf.placeholder(tf.int32, [None, 1], name="rule1_ind")
        rule1_senti = tf.placeholder(tf.int32, [None, 1], name="rule1_senti")
        rule1_rev = tf.ones_like(rule1_senti) - rule1_senti
        rule1_y_pred_p = tf.concat([rule1_rev, rule1_senti], axis=1)
        rule1_full = tf.concat([rule1_ind, rule1_y_pred_p], axis=1)
        # add logic layer
        nclasses = 2
        # Rule_input = cnn.embedded_chars
        Rule_input = cnn.input_x
        rules = [
            FOL_Rule1(nclasses, Rule_input, rule1_full),
        ]
        rule_lambda = [1]  # confidence for the "rule1" rule = 1
        pi_holder = tf.placeholder(tf.float32, [1], name='pi')
        # pi: how percentage listen to teacher loss, starts from lower bound

        logic_nn = LogicNN(
            network=cnn,
            rules=rules,
            rule_lambda=rule_lambda,
            pi=pi_holder,
            C=1.)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(5e-3)
        grads_and_vars = optimizer.compute_gradients(logic_nn.neg_log_liklihood)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        p_loss_summary = tf.summary.scalar("p_loss", cnn.loss)
        p_acc_summary = tf.summary.scalar("p_accuracy", cnn.accuracy)
        q_loss_summary = tf.summary.scalar("q_loss", logic_nn.q_loss)
        q_acc_summary = tf.summary.scalar("q_accuracy", logic_nn.q_accuracy)
        pi_summary = tf.summary.scalar("pi", logic_nn.pi)
        nlld_summmary = tf.summary.scalar("neg_log_liklihood", logic_nn.neg_log_liklihood[()])

        # Train Summaries
        train_summary_op = tf.summary.merge([
            p_loss_summary, p_acc_summary,
            q_loss_summary, q_acc_summary, pi_summary, nlld_summmary,
            grad_summaries_merged,
        ])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([
            p_loss_summary, p_acc_summary,
            q_loss_summary, q_acc_summary, pi_summary, nlld_summmary,
        ])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))
        #
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        #  Use pre-trained word2vec as initializer
        if FLAGS.train_word2vec:
            print("Loading word embeddings ...")
            w2v_path = '../Data/GoogleNews-vectors-negative300.bin'
            w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            embedding_vectors = np.random.uniform(-0.5, 0.5, (len(vocab_processor.vocabulary_), w2v.vector_size))
            for word_, idx_ in vocab_processor.vocabulary_._mapping.items():
                try:
                    embedding_vectors[idx_] = w2v[word_]
                except KeyError as err:
                    pass
            sess.run(cnn.W.assign(embedding_vectors))

        def train_step(x_batch, y_batch, x_fea_batch):
            """
            A single training step
            """
            cur_step = tf.train.global_step(sess, global_step)
            cur_epoch = int(cur_step * 1.0 / FLAGS.batch_size)
            pi = get_pi(cur_iter=cur_epoch,
                        params=FLAGS.pi_params,
                        curve=FLAGS.pi_curve,
                        data_len=x_train.shape[0],)

            feed_dict = {
                logic_nn.network.input_x: x_batch,
                logic_nn.network.input_y: y_batch,
                logic_nn.network.dropout_keep_prob: FLAGS.dropout_keep_prob,
                logic_nn.pi: pi,
                rule1_ind: np.expand_dims(x_fea_batch["rule1_ind"], 1),
                rule1_senti: np.expand_dims(x_fea_batch["rule1_senti"], 1),
            }
            _, step, summaries, neg_log_liklihood, accuracy = sess.run(
                [train_op, global_step, train_summary_op, logic_nn.neg_log_liklihood, logic_nn.network.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # if step % FLAGS.evaluate_every == 0:
            # print("{}: step {}, nlld {:g}, acc {:g}".format(time_str, step, neg_log_liklihood, accuracy))
            # print("{}: step {}, nlld {}, acc {:g}".format(time_str, step, neg_log_liklihood, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, x_fea_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            cur_step = tf.train.global_step(sess, global_step)
            cur_epoch = int(cur_step * 1.0 / FLAGS.batch_size)
            pi = get_pi(cur_iter=cur_epoch,
                        params=FLAGS.pi_params,
                        curve=FLAGS.pi_curve,
                        data_len=x_train.shape[0],)

            feed_dict = {
                logic_nn.network.input_x: x_batch,
                logic_nn.network.input_y: y_batch,
                logic_nn.network.dropout_keep_prob: 1.,
                logic_nn.pi: pi,
                rule1_ind: np.expand_dims(x_fea_batch["rule1_ind"], 1),
                rule1_senti: np.expand_dims(x_fea_batch["rule1_senti"], 1),
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, logic_nn.network.loss, logic_nn.network.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        def get_pi(cur_iter, params=None, curve='exp_arise', data_len=None):
            """
            pi: how percentage listen to teacher loss,
                starts from lower bound
            """
            k, lb = params[0], params[1]
            if curve == 'exp_arise':
                """exponential arise: pi_t = max{1 - k^t, lb}"""
                pi = max([1. - k**(cur_iter*10), float(lb)])
            elif curve == 'exp_decay':
                """exponential decay: pi_t = max{k^t, lb}"""
                pi = max([k**cur_iter, float(lb)])
            elif curve == 'linear_arise':
                """ linear arise : pi_t = t / num_of_steps"""
                num_batches_per_epoch = int((data_len - 1) / FLAGS.batch_size) + 1
                pi = max(cur_iter / (num_batches_per_epoch * FLAGS.num_epochs), lb)
            elif curve == 'linear_decay':
                """ linear decay : pi_t = 1 - t / num_of_steps"""
                num_batches_per_epoch = int((data_len - 1) / FLAGS.batch_size) + 1
                pi = max(1 - cur_iter / (num_batches_per_epoch * FLAGS.num_epochs), lb)
            elif curve == 'constant':
                """ constant pi """
                pi = k
            return pi

        # Batches Generator
        batches = data_helpers.batch_fea_iter(
            x_train, y_train, x_fea_train, FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch, x_fea_batch = batch
            # x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, x_fea_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, x_fea_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
