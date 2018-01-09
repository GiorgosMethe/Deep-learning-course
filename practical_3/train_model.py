from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from convnet import ConvNet
from siamese import Siamese

import cifar10_utils
import cifar10_siamese_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 5000
CHECKPOINT_FREQ_DEFAULT = 10000
PRINT_FREQ_DEFAULT = 1000
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
EXPERIMENT = '/' + 'linear-64-reg-.01'
LOG_DIR_DEFAULT = './logs/cifar10' + EXPERIMENT
CHECKPOINT_DIR_DEFAULT = './checkpoints' + EXPERIMENT

cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

cifar10_siamese = cifar10_siamese_utils.get_cifar10(DATA_DIR_DEFAULT, one_hot=False)

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

    model.wd = None

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_ = model.inference(x)

    with tf.name_scope('loss'):
        loss = model.loss(x_, y_)

    with tf.name_scope('accuracy'):
        accuracy = model.accuracy(x_, y_)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_DEFAULT).minimize(loss)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(LOG_DIR_DEFAULT + '/linear-train', session.graph)
    test_writer = tf.summary.FileWriter(LOG_DIR_DEFAULT + '/linear-test', session.graph)

    saver = tf.train.Saver(tf.global_variables())

    for iteration in range(1, MAX_STEPS_DEFAULT+1):
        model.isTrain = True
        batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
        _, summary = session.run([train_step, merged], feed_dict={x:batch_x, y_:batch_y})
        train_writer.add_summary(summary, iteration)

        if iteration % CHECKPOINT_FREQ_DEFAULT == 0:
            saver.save(session, CHECKPOINT_DIR_DEFAULT + "/linear-model-at-" + str(iteration) + ".ckpt")

        if iteration % EVAL_FREQ_DEFAULT == 0.0:
            _split = 250
            avg_loss, avg_acc = 0.0, 0.0
            model.isTrain = False
            for _iter in range(int(len(cifar10.test.images) / _split)):
                batch_x, batch_y = cifar10.test.images[_iter * _split:((_iter + 1) * _split)],\
                                   cifar10.test.labels[_iter * _split:((_iter + 1) * _split)]

                l, acc = session.run([loss, accuracy], feed_dict={x: batch_x, y_: batch_y})
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
        __, summary = session.run([train_step, merged], feed_dict={x1:batch_x1, x2:batch_x2, y_:batch_labels})

        train_writer.add_summary(summary, iteration)

        if iteration % CHECKPOINT_FREQ_DEFAULT == 0:
            saver.save(session, CHECKPOINT_DIR_DEFAULT + "/siamese-model-at-" + str(iteration) + ".ckpt")

        if iteration % EVAL_FREQ_DEFAULT == 0.0:
            avg_loss = 0.0
            for i in range(len(val_set)):
                batch_x1, batch_x2, batch_labels = val_set[i][0], val_set[i][1], val_set[i][2]
                l = session.run(loss, feed_dict={x1: batch_x1, x2: batch_x2, y_: batch_labels})
                avg_loss += l
            avg_loss /= float(len(val_set))
            summary = tf.Summary()
            summary.value.add(tag='loss', simple_value=avg_loss)
            test_writer.add_summary(summary, iteration)
            print(iteration, avg_loss)
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

        with tf.device('/cpu:0'):
            model = ConvNet()
            model.isTrain = False
            model.wd = None

            x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])

            x_ = model.inference(x)

            features_flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/flatten/h_f_flatten:0")
            features_fc1 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc1/h_f_fc1:0")
            features_fc2 = tf.get_default_graph().get_tensor_by_name("ConvNet/fc2/h_f_fc2:0")

            init = tf.initialize_all_variables()
            session = tf.Session()
            session.run(init)

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(session, CHECKPOINT_DIR_DEFAULT + "/linear-model-at-" + str(MAX_STEPS_DEFAULT) + ".ckpt")

            batch_x, batch_y = cifar10.test.images, cifar10.test.labels
            x_, features_flatten, features_fc1, features_fc2 = session.run([x_, features_flatten, features_fc1, features_fc2], feed_dict={x: batch_x, y_: batch_y})

        '''
        VISUALIZATION
        '''
        if 0:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(features_flatten)
            plt.figure()
            colors = cm.rainbow(np.linspace(0, 1, len(batch_y[0])))
            sizes = np.random.random_integers(25, 50, size=len(batch_y[0]))
            for i in range(len(batch_y[0])):
                index = np.where(np.argmax(batch_y, axis=1) == i)
                plt.scatter(tsne_result[index,0], tsne_result[index, 1], c=colors[i], s=sizes[i], label=i)
            plt.legend(numpoints=1, fontsize=8)
            plt.savefig("visualization-linear-features_flatten.pdf", bbox_inches = 'tight')

            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(features_fc1)
            plt.figure()
            for i in range(len(batch_y[0])):
                index = np.where(np.argmax(batch_y, axis=1) == i)
                plt.scatter(tsne_result[index,0], tsne_result[index, 1], c=colors[i], s=sizes[i], label=i)
            plt.legend(numpoints=1, fontsize=8)
            plt.savefig("visualization-linear-features_fc1.pdf", bbox_inches = 'tight')

            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(features_fc2)
            plt.figure()
            for i in range(len(batch_y[0])):
                index = np.where(np.argmax(batch_y, axis=1) == i)
                plt.scatter(tsne_result[index,0], tsne_result[index, 1], c=colors[i], s=sizes[i], label=i)
            plt.legend(numpoints=1, fontsize=8)
            plt.savefig("visualization-linear-features_fc2.pdf", bbox_inches = 'tight')

        '''
        ONE-VS-REST CLASSIFIER
        '''
        if 0:
            train_set = range(int(features_flatten.shape[0]*0.8))
            test_set = range(train_set[-1]+1, features_flatten.shape[0])

            classif = OneVsRestClassifier(SVC(kernel='linear'))
            classif.fit(features_flatten[train_set], np.argmax(batch_y[train_set], axis=1))
            print("Accuracy with flatten:",
                  classif.score(features_flatten[test_set], np.argmax(batch_y[test_set], axis=1)))

            classif = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=8)
            classif.fit(features_fc1[train_set], np.argmax(batch_y[train_set], axis=1))
            print("Accuracy with features_fc1:",
                  classif.score(features_fc1[test_set], np.argmax(batch_y[test_set], axis=1)))

            classif = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=8)
            classif.fit(features_fc2[train_set], np.argmax(batch_y[train_set], axis=1))
            print("Accuracy with features_fc2:",
                  classif.score(features_fc2[test_set], np.argmax(batch_y[test_set], axis=1)))

    elif FLAGS.train_model == 'siamese':

        with tf.device('/cpu:0'):

            model = Siamese()

            x1 = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

            x1_ = model.inference(x1, reuse=None)

            features = tf.get_default_graph().get_tensor_by_name("ConvNet/l_2-norm:0")

            init = tf.initialize_all_variables()
            session = tf.Session()
            session.run(init)

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(session, CHECKPOINT_DIR_DEFAULT + "/siamese-model-at-" + str(20000) + ".ckpt")

            batch_x1, batch_y = cifar10.test.images, np.argmax(cifar10.test.labels, axis=1)

            features = session.run(features, feed_dict={x1: batch_x1})

        '''
            VISUALIZATION
        '''
        if 0:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(features)
            plt.figure()
            colors = cm.rainbow(np.linspace(0, 1, 10))
            sizes = np.random.random_integers(25, 50, size=10)
            for i in range(10):
                index = np.where(batch_y == i)
                plt.scatter(tsne_result[index, 1], tsne_result[index, 0], c=colors[i], s=sizes[i], label=i)
            plt.legend(numpoints=1, fontsize=8)
            plt.savefig("visualization-siamese-3.pdf", bbox_inches='tight')
            plt.show()

        '''
            One-vs-All Classifiers
        '''
        if 0:
            train_set = range(int(features.shape[0] * 0.8))
            test_set = range(train_set[-1] + 1, features.shape[0])

            classif = OneVsRestClassifier(SVC(kernel='linear'))
            classif.fit(features[train_set], batch_y[train_set])
            print("Accuracy with l_2-norm:",
                  classif.score(features[test_set], batch_y[test_set]))

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
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
