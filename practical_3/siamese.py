from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet') as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            with tf.variable_scope('conv1', reuse=reuse) as scope:
                w_conv1 = tf.get_variable('weights',
                                          shape=[5, 5, 3, 64],
                                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

                b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]), name='biases')

                h_conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="h_f_1")

            with tf.variable_scope('conv2', reuse=reuse) as scope:
                w_conv2 = tf.get_variable('weights',
                                          shape=[5, 5, 64, 64],
                                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

                b_conv2 = tf.Variable(tf.constant(0.0, shape=[64]), name='biases')

                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name="h_f_2")

            with tf.variable_scope('flatten', reuse=reuse) as scope:
                flatten = tf.reshape(h_pool2, shape=[-1,
                                                     int(h_pool2.get_shape()[1] *
                                                         h_pool2.get_shape()[2] *
                                                         h_pool2.get_shape()[3])], name="h_f_flatten")

            with tf.variable_scope('fc1', reuse=reuse) as scope:
                w_fc1 = tf.get_variable('weights',
                                        shape=[flatten.get_shape()[1], 384],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

                b_fc1 = tf.Variable(tf.constant(0.0, shape=[384]), name='biases')

                h_fc1 = tf.nn.relu(tf.matmul(flatten, w_fc1) + b_fc1, name="h_f_fc1")

            with tf.variable_scope('fc2', reuse=reuse) as scope:
                w_fc2 = tf.get_variable('weights',
                                        shape=[384, 192],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

                b_fc2 = tf.Variable(tf.constant(0.0, shape=[192]), name='biases')

                h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2, name="h_f_fc2")

            l2_out = tf.nn.l2_normalize(h_fc2, dim=1, name="l_2-norm")
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        l2_dist = tf.sqrt(tf.reduce_sum(tf.square(channel_1 - channel_2)))
        contrast = tf.maximum(0.0, margin - l2_dist)
        loss = tf.reduce_sum(label * l2_dist + (1.0 - label) * contrast)
        ########################
        # END OF YOUR CODE    #
        ########################
        return loss
