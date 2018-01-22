"""
First Order Logic (FOL) rules
"""

import numpy as np
import tensorflow as tf


class FOL(object):
    """ First Order Logic (FOL) rules """

    def __init__(self, K, input, fea, true_label):
        """ Initialize

        : param K: the number of classes, type: int
        """
        self.K = K
        self.input = input
        self.fea = fea
        self.true_label = true_label
        # Record the data relevance (binary)
        self.conds = self.conditions(self.input, self.fea)

    def conditions(self, X, F):
        # X = input ; F = rule feature
        return tf.map_fn(lambda i: self.condition_single(i[0], i[1]), (X, F), dtype=tf.float32)
    
    '''
    def distribution_helper_helper(self, x, f):
        output = tf.map_fn(
            lambda k: self.value_single(x, tf.cast(k, tf.float32), f),
            tf.range(self.K),
            dtype=tf.float32
        )
        return output
    '''

    def distribution_helper(self, w, X, F, true_label, conds, batch_size):
        # nx = int(batch_size)
        # distr = tf.constant(1., shape=[nx, self.K], dtype=tf.float32)
        X_f = tf.cast(X, tf.float32)
        Y = tf.cast(tf.argmax(true_label, axis=1), tf.float32)
        distr = tf.map_fn(lambda i: self.value_single(i[0], i[1], i[2]), (X_f, Y, F) , dtype=tf.float32)
        return distr   

    # ---------------------------------------------------------------------------------
    """
    Interface function of logic constraints

    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow

    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """

    def log_distribution(self, w, X=None, F=None, true_label=None, batch_size = 64,config={}):
        """
        w: C * lambda_l ; X: input word vector ; F: input feature(rule predictions)
        
        Return an nxk matrix, each element
        = - (w:C*lambda) * (1 - r(X_i, Y_i)) ,if X_i is a grounding of the rule
        = 1 ,otherwise
        """
        if F is None:
            X, F, conds, true_label = self.input, self.fea, self.conds, self.true_label
        else:
            conds = self.conditions(X, F)
        distr = self.distribution_helper(w, X, F, true_label, conds, batch_size)
        return tf.cast(tf.where(tf.equal(conds, 1.), -w*distr, tf.ones(tf.shape(distr))), dtype=tf.float32)
    
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


class FOL_Rule1(FOL):
    #""" x=x1_love_x2 => { ~y => pos(x2) AND pos(x2) => ~y } """
    """ x=x1_love_x2 => { pos(x2) => ~y } """

    def __init__(self, K, input, fea, true_label):
        """ Initialize

        :param K: the number of classes, type:int

        :param fea: symbolic feature tensor, of shape 3
                    fea[0]   : 1 if x=x1_love_x2, 0 otherwise
                    fea[1:2] : classifier.predict_p(x_2)
        """
        assert K == 2
        super(FOL_Rule1, self).__init__(K, input, fea, true_label)

    # ---------------------------------------------------------------------------------
    """
    Rule-specific functions

    """
    def condition_single(self, x, f):
        return tf.cast(tf.equal(f[0], 1.), tf.float32)

    def value_single(self, x, y, f):
        #ret = tf.reduce_mean([
        #    tf.minimum(1. - (1. - y) + f[2], 1.),
        #    tf.minimum(1. - f[2] + (1. - y), 1.)
        #])
        ret = tf.minimum(1. - f[2] + (1. - y), 1.)
        return tf.cast(
            tf.where(tf.equal(self.condition_single(x, f), 1.), 1 - ret, 1.),
            dtype=tf.float32
        )

    '''  
    def log_distribution(self, w, X=None, F=None, true_label=None, batch_size = 64):
        if F is None:
            X, F = self.input, self.fea
        F_mask = F[:, 0]
        F_fea = F[:, 1:]
        # y = 0
        distr_y0 = w * F_mask * F_fea[:, 0] #F[1]
        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1] #F[2]
        distr_y0 = tf.expand_dims(distr_y0, -1)
        distr_y1 = tf.expand_dims(distr_y1, -1)
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr
    '''

##==============================================================================

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
        distr_y0 = w * F_mask * F_fea[:, 0] #F[1]
        # y = 1
        distr_y1 = w * F_mask * F_fea[:, 1] #F[2]
        distr_y0 = tf.expand_dims(distr_y0, -1)
        distr_y1 = tf.expand_dims(distr_y1, -1)
        distr = tf.concat([distr_y0, distr_y1], axis=1)
        return distr


