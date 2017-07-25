import numpy as np
import tensorflow as tf


class LogicNN(object):  # LogicNN takes a network as parameter
    def __init__(self, rng, input, network, rules=[], rule_lambda=[], pi=0., C=1.):
        """
        :param input: symbolic image tensor, of shape image_shape
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
        dropout_q_y_given_x = self.network.h_drop * 1.0  # self.network.h_drop  = self.network.dropout_p_y_given_x
        q_y_given_x = self.network.scores * 1.0  # self.network.scores  = self.network.p_y_given_x
        # combine rule constraints
        distr = self.calc_rule_constraints()
        q_y_given_x *= distr
        dropout_q_y_given_x *= distr

        # normalize (dropout_q_y_given_x)
        n = int(self.input.get_shape()[0])
        n_dropout_q_y_given_x = dropout_q_y_given_x / tf.reshape(tf.reduce_sum(dropout_q_y_given_x, axis=1), [n, 1])
        n_q_y_given_x = q_y_given_x / tf.reshape(tf.reduce_sum(q_y_given_x, axis=1), [n, 1])
        self.dropout_q_y_given_x = tf.stop_gradient(n_dropout_q_y_given_x)
        self.q_y_given_x = tf.stop_gradient(n_q_y_given_x)

        # compute prediction as class whose probability is maximal in symbolic form
        self.q_y_pred = tf.argmax(q_y_given_x, axis=1)
        self.p_y_pred = self.network.predictions

        # collect all learnable parameters
        # self.params_p = self.network.params

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

    def dropout_negative_log_likelihood(self, y):
        nlld = (1.0 - self.pi) * self.network.dropout_negative_log_likelihood(y)
        nlld += self.pi * self.network.soft_dropout_negative_log_likelihood(self.dropout_q_y_given_x)
        return nlld

    def negative_log_likelihood(self, y):
        nlld = (1.0 - self.pi) * self.network.loss
        nlld += self.pi * self.network.soft_negative_log_likelihood(self.q_y_given_x)
        return nlld

    def errors(self, y):  # return average mistakes by q / p
        # check if y has same dimension of y_pred
        if y.ndim != self.q_y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.q_y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.q_y_pred, y)), T.mean(T.neq(self.p_y_pred, y))
        else:
            raise NotImplementedError()

    # return LogicNN's q/p argmax predictions
    def predict(self, new_data_to_network, new_data, new_rule_fea):
        q_y_pred_p, p_y_pred_p = self.predict_p(new_data_to_network, new_data, new_rule_fea)
        q_y_pred = T.argmax(q_y_pred_p, axis=1)
        p_y_pred = T.argmax(p_y_pred_p, axis=1)
        return q_y_pred, p_y_pred

    # return LogicNN's q/p prob. predictions
    def predict_p(self, new_data_to_network, new_data, new_rule_fea):
        p_y_pred_p = self.network.predict_p(new_data_to_network)
        q_y_pred_p = p_y_pred_p * self.calc_rule_constraints(new_data=new_data, new_rule_fea=new_rule_fea)
        return q_y_pred_p, p_y_pred_p
