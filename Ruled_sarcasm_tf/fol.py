"""
First Order Logic (FOL) rules
"""

import numpy as np
import tensorflow as tf


class FOL(object):
    """ First Order Logic (FOL) rules """

    def __init__(self, K, input, fea):
        """ Initialize

        : param K: the number of classes, type: int
        """
        self.K = K
        self.input = input
        self.fea = fea
        # Record the data relevance (binary)
        self.conds = self.conditions(self.input, self.fea)

    def conditions(self, X, F):
        return tf.map_fn(lambda i: self.condition_single(i[0], i[1]), (X, F), dtype=tf.float32)

    def distribution_helper_helper(self, x, f):
        return tf.map_fn(
            lambda k: self.value_single(x, k, f),
            tf.range(0, self.K))

    def distribution_helper(self, w, X, F, conds):
        nx = int(X.get_shape()[0])
        distr = tf.constant(1., shape=[nx, self.K], dtype=tf.float32)
        def _f(_, item):
            c, x, f, d = item
            return tf.select(tf.equal(c, 1.), self.distribution_helper_helper(x, f), d)
        distr = tf.scan(
            _f,
            (conds, X, F, distr),
        )
        distr = tf.map_fn(
            lambda d: -w * (tf.reduce_min(d, keepdims=True) - d),  # relative value w.r.t the minimum
            distr)
        return distr
    # ---------------------------------------------------------------------------------
    """
    Interface function of logic constraints

    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow

    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None, config={}):
        """ Return an nxK matrix with the (i,c)-th term
        = - w * (1 - r(X_i, y_i=c))
               if X_i is a grounding of the rule
        = 1    otherwise
        """
        if F is None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(X, F)
        log_distr = self.distribution_helper(w, X, F, conds)
        return log_distr
    # ---------------------------------------------------------------------------------
    """
    Rule-specific functions to be overloaded
    """

    def condition_single(self, x, f):
        """ True if x satisfies the condition """
        return tf.zeros([1], dtype=tf.float32)

    def value_single(self, x, y, f):
        """ value = r(x,y) """
        return tf.ones([1], dtype=tf.float32)


class FOL_But(FOL):
    """ x=x1_but_x2 => { y => pred(x2) AND pred(x2) => y } """

    def __init__(self, K, input, fea):
        """ Initialize

        :param K: the number of classes, type: int

        :param fea: symbolic feature tensor, of shape 3
                    fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                    fea[1:2] : classifier.predict_p(x_2)
        """
        assert K == 2
        super(FOL_But, self).__init__(K, input, fea)
    # ---------------------------------------------------------------------------------
    """
    Rule-specific functions
    """

    def condition_single(self, x, f):
        return tf.cast(tf.equal(f[0], 1.), tf.float32)

    def value_single(self, x, y, f):
        ret = tf.reduce_mean([tf.minimum([1. - y + f[2], 1.]), tf.minimum([1. - f[2] + y, 1.])])
        ret = tf.cast(ret, tf.float32)
        return tf.cast(
            tf.select(tf.equal(self.condition_single(x, f), 1.), ret, 1.),
            dtype=tf.float32)
    # ---------------------------------------------------------------------------------
    """
    Efficient version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None):
        if F is None:
            X, F = self.input, self.fea
        F_mask = F[:, 0]
        F_fea = F[:, 1:]
        # y = 0
        distr_y0 = w * F_mask * F_fea[:, 0]
        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1]
        distr_y0 = tf.expand_dims(distr_y0, -1)
        distr_y1 = tf.expand_dims(distr_y1, -1)
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr


class FOL_Rule1(FOL):
    #""" x=x1_love_x2 => { ~y => pos(x2) AND pos(x2) => ~y } """
    """ x=x1_love_x2 => { pos(x2) => ~y } """

    def __init__(self, K, input, fea):
        """ Initialize

        :param K: the number of classes, type:int

        :param fea: symbolic feature tensor, of shape 3
                    fea[0]   : 1 if x=x1_love_x2, 0 otherwise
                    fea[1:2] : classifier.predict_p(x_2)
        """
        assert K == 2
        super(FOL_Rule1, self).__init__(K, input, fea)

    # ---------------------------------------------------------------------------------
    """
    Rule-specific functions

    """
    def condition_single(self, x, f):
        return tf.cast(tf.equal(f[0], 1), tf.float32)

    def value_single(self, x, y, f):
        #ret = tf.reduce_mean([
        #    tf.minimum([1. - (1 - y) + f[2], 1.]),
        #    tf.minimum([1. - f[2] + (1 - y), 1.])
        #])
        ret = tf.minimum([1. - f[2] + (1 - y), 1.])
        ret = tf.cast(ret, tf.float32)
        return tf.cast(
            tf.select(tf.equal(self.condition_single(x, f), 1.), ret, 1.),
            dtype=tf.float32
        )
