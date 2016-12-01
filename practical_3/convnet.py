from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import regularizers
import numpy as np
from tensorflow.models.image.cifar10 import cifar10



class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            with tf.variable_scope('conv1') as scope:
                W_conv1 = tf.get_variable('weights', shape=[5, 5, 3, 64],
                                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                          regularizer=regularizers.l2_regularizer(0.001))
                b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]), name='biases')

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            print(h_pool1.get_shape())

            with tf.variable_scope('conv2') as scope:
                W_conv2 = tf.get_variable('weights', shape=[5, 5, 64, 64],
                                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                          regularizer=regularizers.l2_regularizer(0.001))
                b_conv2 = tf.Variable(tf.constant(0.0, shape=[64]), name='biases')

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            print(h_pool2.get_shape())

            with tf.variable_scope('flatten') as scope:
                flatten = tf.reshape(h_pool2, [128, -1])

            with tf.variable_scope('fc1') as scope:
                W_fc1 = tf.get_variable('weights',
                                        shape=[128, 384],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                        regularizer=regularizers.l2_regularizer(0.001))
                b_fc1 = tf.Variable(tf.constant(0.0, shape=[384]), name='biases')

            h_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1)

            print(h_fc1.get_shape())

            with tf.variable_scope('fc2') as scope:
                W_fc2 = tf.get_variable('weights',
                                        shape=[384, 192],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                        regularizer=regularizers.l2_regularizer(0.001))
                b_fc2 = tf.Variable(tf.constant(0.0, shape=[192]), name='biases')

            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            print(h_fc2.get_shape())

            with tf.variable_scope('fc3') as scope:
                W_fc3 = tf.get_variable('weights', shape=[192, 10],
                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                        regularizer=regularizers.l2_regularizer(0.001))
                b_fc3 = tf.Variable(tf.constant(0.0, shape=[10]), name='biases')

                h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

            print(h_fc3.get_shape())

            ########################
            # END OF YOUR CODE    #
            ########################
        return h_fc3

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ########################
        # END OF YOUR CODE    #
        ########################
        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE   #
        ########################
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
        loss = cross_entropy_mean
        ########################
        # END OF YOUR CODE     #
        ########################
        return loss
