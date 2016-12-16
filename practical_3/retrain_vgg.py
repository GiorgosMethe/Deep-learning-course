from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import vgg
from convnet import ConvNet

import tensorflow as tf
import numpy as np

import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 10000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 100

EXPERIMENT = '/' + 'vgg16-refine-100'
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10' + EXPERIMENT
CHECKPOINT_DIR_DEFAULT = './checkpoints' + EXPERIMENT

cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    model = ConvNet()

    vgg.load_weights(vgg.VGG_FILE)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    refine = tf.placeholder(tf.bool)

    pool5, assign_ops = vgg.load_pretrained_VGG16_pool5(x)
    pool5 = tf.cond(refine, lambda : pool5, lambda : tf.stop_gradient(pool5))


    #############
    ## Extra layers
    #############
    with tf.variable_scope('flatten') as scope:
        flatten = tf.reshape(pool5, shape=[-1, 512], name="h_f_flatten")

    with tf.variable_scope('fc1') as scope:
        w_fc1 = tf.get_variable('weights',
                                shape=[flatten.get_shape()[1], 384],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))
        b_fc1 = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.0))

    h_fc1 = tf.nn.relu(tf.matmul(flatten, w_fc1) + b_fc1, name="h_f_fc1")

    with tf.variable_scope('fc2') as scope:
        w_fc2 = tf.get_variable('weights',
                                shape=[384, 192],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

    b_fc2 = tf.get_variable('biases', shape=[192], initializer=tf.constant_initializer(0.0))

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2, name="h_f_fc2")

    with tf.variable_scope('fc3') as scope:
        w_fc3 = tf.get_variable('weights',
                                shape=[192, 10],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001))

        b_fc3 = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0))

    #############
    ## Extra layers
    #############

    x_ = tf.matmul(h_fc2, w_fc3) + b_fc3

    with tf.name_scope('loss'):
        loss = model.loss(x_, y_)

    with tf.name_scope('accuracy'):
        accuracy = model.accuracy(x_, y_)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_DEFAULT).minimize(loss)

    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    # Assign ops for weight restoration
    session.run(assign_ops)

    tf.scalar_summary('accuracy', accuracy)
    tf.scalar_summary('loss', loss)
    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/linear-train', session.graph)
    test_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/linear-test', session.graph)

    saver = tf.train.Saver(tf.all_variables())

    for iteration in range(1, MAX_STEPS_DEFAULT + 1):
        model.isTrain = True
        batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
        _, summary, l, acc = session.run([train_step, merged, loss, accuracy],
                                         feed_dict={refine: (iteration > REFINE_AFTER_K_STEPS_DEFAULT),
                                                    x: batch_x, y_: batch_y})

        train_writer.add_summary(summary, iteration)

        if iteration % CHECKPOINT_FREQ_DEFAULT == 0:
            saver.save(session, CHECKPOINT_DIR_DEFAULT + "/linear-model-at-" + str(iteration) + ".ckpt")

        if iteration % EVAL_FREQ_DEFAULT == 0.0 or iteration == 1:
            _split = 250
            avg_loss, avg_acc = 0.0, 0.0
            model.isTrain = False
            for _iter in range(int(len(cifar10.test.images) / _split)):
                batch_x, batch_y = cifar10.test.images[_iter * _split:((_iter + 1) * _split)], \
                                   cifar10.test.labels[_iter * _split:((_iter + 1) * _split)]

                l, acc = session.run([loss, accuracy], feed_dict={refine: 0, x: batch_x, y_: batch_y})
                avg_loss += l
                avg_acc += acc
            avg_loss /= float(len(cifar10.test.images)) / float(_split)
            avg_acc /= float(len(cifar10.test.images)) / float(_split)
            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=avg_loss)
            summary.value.add(tag='accuracy', simple_value=avg_acc)
            test_writer.add_summary(summary, iteration)
            print(iteration, avg_loss, avg_acc)

    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
