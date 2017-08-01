import numpy as np
import tensorflow as tf


class LogicNN(object):
    def __init__(self, rng, input, network, rules=[], rule_lambda=[], pi=0., C=1.):
        """
        :param input: symbolic image tensor, of shape image_shape
        :param network: student network
        """
        self.input = input
        self.network = network
        self.rules = rules
        self.rule_lambda = tf.constant(rule_lambda, dtype=tf.float32, name='rule_lambda')
        self.ones = tf.ones([len(rules)], name='ones', dtype=tf.float32)
        self.pi = tf.constant(pi, shape=[1], name='pi', dtype=tf.float32)
        self.C = C

        ## q(y|x)
        # dropout_p_y_given_x: output of a LogisticRegression Layer (of a network)
        dropout_q_y_given_x = self.network.h_drop_p * 1.0  # self.network.h_drop  = self.network.dropout_p_y_given_x
        q_y_given_x = self.network.predict_p * 1.0  # self.network.predict_p  = self.network.p_y_given_x
        # combine rule constraints
        distr = self.calc_rule_constraints(new_data=new_data, new_rule_fea=new_rule_fea)
        # TODO: calc_rule_constraints() lack of input argument

        q_y_given_x *= distr
        dropout_q_y_given_x *= distr

        # normalize (dropout_q_y_given_x)
        n = int(self.input.get_shape()[0])
        n_dropout_q_y_given_x = dropout_q_y_given_x / tf.reshape(tf.reduce_sum(dropout_q_y_given_x, axis=1), [n, 1])
        n_q_y_given_x = q_y_given_x / tf.reshape(tf.reduce_sum(q_y_given_x, axis=1), [n, 1])
        self.dropout_q_y_given_x = tf.stop_gradient(n_dropout_q_y_given_x)
        self.q_y_given_x = tf.stop_gradient(n_q_y_given_x)


        # collect all learnable parameters
        # self.params_p = self.network.params

        q_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.network.scores, labels=self.q_y_given_x)
        self.q_loss = tf.reduce_mean(q_loss)

        self.neg_log_liklihood = (1.0 - self.pi) * self.network.loss
        self.neg_log_liklihood += self.pi * self.q_loss

        drop_q_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.network.h_drop, labels=self.q_y_given_x)
        self.drop_q_loss = tf.reduce_mean(drop_q_loss)

        self.drop_neg_log_liklihood = (1.0 - self.pi) * self.network.drop_loss
        self.drop_neg_log_liklihood += self.pi * self.drop_q_loss


        # compute prediction as class whose probability is maximal in symbolic form
        # return LogicNN's q/p argmax predictions
        self.q_y_pred = tf.argmax(q_y_given_x, axis=1)
        self.p_y_pred = self.network.predictions

        # return LogicNN's q/p prob. predictions
        self.p_y_pred_p = self.network.predict_p
        self.q_y_pred_p = p_y_pred_p * self.calc_rule_constraints(new_data=new_data, new_rule_fea=new_rule_fea)
        # TODO: calc_rule_constraints() lack of input argument

        q_correct_predictions = tf.equal(self.q_y_pred, tf.argmax(self.network.input_y, 1))
        self.q_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="q_accuracy")

        
    # methods for LogicNN
    def calc_rule_constraints(self, new_data=None, new_rule_fea=None):
        if new_rule_fea is None:
            new_rule_fea = [None] * len(self.rules)
        distr_all = tf.zeros(shape=[1], dtype=tf.float32)  # will be broadcast
        for i, rule in enumerate(self.rules):
            distr = rule.log_distribution(self.C * self.rule_lambda[i], new_data, new_rule_fea[i])
            distr_all += distr
        distr_all += distr
        #
        distr_y0 = distr_all[:, 0]
        distr_y0 = tf.expand_dims(distr_y0, -1)
        distr_y0_copies = tf.tile(distr_y0, [1, int(distr_all.get_shape()[1])])
        distr_all -= distr_y0_copies
        distr_all = tf.maximum(tf.minimum(distr_all, 60.), -60.)  # truncate to avoid over-/under-flow
        return tf.exp(distr_all)

    # def set_pi(self, new_pi):
    #     self.pi.set_value(new_pi)

    # def get_pi(self):
    #     return self.pi.get_value()
