import numpy as np
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, 
                 num_filters, vocab_size, l2_reg_lambda=0.0, train_w2v=False):
        
        # Placeholders for input, output and dropout
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        if train_w2v:
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            # Embedding layer
            with tf.device('/cpu'), tf.variable_scope("embedding"):
                self.W = tf.get_variable(
                    "W",
                    initializer=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
                # size:  [None, sequence_length, embedding_size, 1]
        else:
            self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
            self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

                # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv1-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable("W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_filters]))
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")  # size: [None, 49-3+1, 1,128]
                # Apply batch_normalization
                #conv_bn = tf.contrib.layers.batch_norm(conv, center=True, scale=False, scope='BN')
                # Apply nonlinearity
                #h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), name="relu")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, filter_size, 1, 1],
                    #ksize=[1, sequence_length - filter_size + 1 , 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                #pooled_outputs.append(pooled)
            with tf.variable_scope("conv2-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 1, num_filters, num_filters]
                W = tf.get_variable("W", initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_filters]))
                conv2 = tf.nn.conv2d(
                    pooled,  # size: [None, 45, 1, 128]
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply batch_normalization
                #conv_bn = tf.contrib.layers.batch_norm(conv2, center=True, scale=False, scope='BN')
                # Apply nonlinearity
                #h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), name="relu")  # size: [None, 43, 1, 128]
                h = tf.nn.relu(tf.nn.bias_add(conv2, b), name="relu")  # size: [None, 43, 1, 128]
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(  # size: [None, 43, 1, 128]
                    h,
                    ksize=[1, h.get_shape()[1], 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
            
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name='h_drop')
            self.h_drop_p = tf.nn.softmax(self.h_drop, dim=-1, name='h_drop_p')

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", initializer=tf.constant(0.1, shape=[num_classes]))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predict_p = tf.nn.softmax(self.scores, dim=-1, name='predict_p')
            self.predictions = tf.argmax(self.predict_p, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("drop_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.h_drop, labels=self.input_y)
            self.drop_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
