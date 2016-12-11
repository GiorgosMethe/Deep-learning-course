from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
from convnet import ConvNet
from siamese import Siamese

import cifar10_utils
import cifar10_siamese_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

cifar10_siamese = cifar10_siamese_utils.get_cifar10(DATA_DIR_DEFAULT,
                                                     one_hot=False)

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
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    model = ConvNet()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_ = model.inference(x)

    with tf.name_scope('loss'):
        loss = model.loss(x_, y_)

    with tf.name_scope('accuracy'):
        accuracy = model.accuracy(x_, y_)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_DEFAULT).minimize(loss)

    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    tf.scalar_summary('accuracy', accuracy)
    tf.scalar_summary('loss', loss)
    tf.histogram_summary('logits', x_)
    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/linear-train', session.graph)
    test_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/linear-test', session.graph)

    saver = tf.train.Saver(tf.all_variables())

    for iteration in range(1, MAX_STEPS_DEFAULT+1):
        batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
        __, summary, l, acc = session.run([train_step, merged, loss, accuracy], feed_dict={x:batch_x, y_:batch_y})
        train_writer.add_summary(summary, iteration)

        if iteration % CHECKPOINT_FREQ_DEFAULT == 0:
            saver.save(session, CHECKPOINT_DIR_DEFAULT + "/linear-model-at-" + str(iteration) + ".ckpt")

        if iteration % EVAL_FREQ_DEFAULT == 0.0:
            _split = 250
            avg_loss, avg_acc = 0.0, 0.0
            for _iter in range(int(len(cifar10.test.images) / _split)):
                batch_x, batch_y = cifar10.test.images[_iter * _split:((_iter + 1) * _split)],\
                                   cifar10.test.labels[_iter * _split:((_iter + 1) * _split)]

                __, summary, l, acc = session.run([train_step, merged, loss, accuracy], feed_dict={x: batch_x, y_: batch_y})
                avg_loss += l
                avg_acc += acc
            avg_loss /= float(len(cifar10.test.images)) / float(_split)
            avg_acc /= float(len(cifar10.test.images)) / float(_split)
            print(iteration, avg_loss, avg_acc)
            test_writer.add_summary(summary, iteration)
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    val_set = cifar10_siamese_utils.create_dataset(cifar10_siamese,
                                                   num_tuples=600,
                                                   batch_size=BATCH_SIZE_DEFAULT,
                                                   fraction_same=0.2)

    model = Siamese()

    x1 = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    x2 = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None,])

    x1_ = model.inference(x1, reuse=None)
    x2_ = model.inference(x2, reuse=True)

    with tf.name_scope('loss'):
        loss = model.loss(x1_, x2_, y_, margin=1.0)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_DEFAULT).minimize(loss)

    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    tf.scalar_summary('loss', loss)
    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/siamese-train', session.graph)
    test_writer = tf.train.SummaryWriter(LOG_DIR_DEFAULT + '/siamese-test', session.graph)

    saver = tf.train.Saver(tf.all_variables())

    for iteration in range(1, MAX_STEPS_DEFAULT+1):
        batch_x1, batch_x2, batch_labels = cifar10_siamese.train.next_batch(BATCH_SIZE_DEFAULT)
        __, summary, l = session.run([train_step, merged, loss], feed_dict={x1:batch_x1, x2:batch_x2, y_:batch_labels})

        train_writer.add_summary(summary, iteration)

        print(iteration, l)

        if iteration % CHECKPOINT_FREQ_DEFAULT == 0:
            saver.save(session, CHECKPOINT_DIR_DEFAULT + "/siamese-model-at-" + str(iteration) + ".ckpt")

        if iteration % EVAL_FREQ_DEFAULT == 0.0:
            avg_loss = 0.0
            for i in range(len(val_set)):
                batch_x1, batch_x2, batch_labels = val_set[i][0], val_set[i][1], val_set[i][2]
                __, summary, l = session.run([train_step, merged, loss], feed_dict={x1: batch_x1, x2: batch_x2, y_: batch_labels})
                avg_loss += l
            avg_loss /= float(len(val_set))
            print(iteration, avg_loss)
            test_writer.add_summary(summary, iteration)

    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    if FLAGS.train_model == 'linear':
        model = ConvNet()

        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        x_ = model.inference(x)

        init = tf.initialize_all_variables()
        session = tf.Session()
        session.run(init)

        new_saver = tf.train.import_meta_graph(CHECKPOINT_DIR_DEFAULT + "/model-at-" + str(MAX_STEPS_DEFAULT) + ".ckpt.meta")
        new_saver.restore(session, CHECKPOINT_DIR_DEFAULT + "/model-at-" + str(MAX_STEPS_DEFAULT) + ".ckpt")

        batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
        x_ = session.run(x_, feed_dict={x: batch_x, y_: batch_y})
        print(x_)

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

    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = False,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
