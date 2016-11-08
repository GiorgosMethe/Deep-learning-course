# Import necessary standard python packages 
import numpy as np
import matplotlib.pyplot as plt

# Setting configuration for matplotlib
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 20

from uva_code.cifar10_utils import get_cifar10_raw_data, preprocess_cifar10_data
from uva_code.solver import Solver
from uva_code.losses import SoftMaxLoss, CrossEntropyLoss, HingeLoss
from uva_code.layers import LinearLayer, ReLULayer, SigmoidLayer, TanhLayer, SoftMaxLayer, ELULayer
from uva_code.models import Network
from uva_code.optimizers import SGD

# Get raw CIFAR10 data. For Unix users the script to download CIFAR10 dataset (get_cifar10.sh).
# Try to run script to download the data. It should download tar archive, untar it and then remove it. 
# If it is doesn't work for some reasons (like Permission denied) then manually download the data from 
# http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and extract it to cifar-10-batches-py folder inside 
# cifar10 folder.

X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = get_cifar10_raw_data()

#Checking shapes, should be (50000, 32, 32, 3), (50000, ), (10000, 32, 32, 3), (10000, )
print("Train data shape: {0}").format(str(X_train_raw.shape))
print("Train labels shape: {0}").format(str(Y_train_raw.shape))
print("Test data shape: {0}").format(str(X_test_raw.shape))
print("Test labels shape: {0}").format(str(Y_test_raw.shape))

# Visualize CIFAR10 data
samples_per_class = 10
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_classes = len(classes)


# Normalize CIFAR10 data by subtracting the mean image. With these data you will work in the rest of assignment.
# The validation subset will be used for tuning the hyperparameters.
X_train, Y_train, X_val, Y_val, X_test, Y_test = preprocess_cifar10_data(X_train_raw, Y_train_raw, 
                                                                                 X_test_raw, Y_test_raw, num_val = 1000)

#Checking shapes, should be (49000, 3072), (49000, ), (1000, 3072), (1000, ), (10000, 3072), (10000, ) 
print "Train data shape: {0}".format(str(X_train.shape))
print "Train labels shape: {0}".format(str(Y_train.shape))
print "Val data shape: {0}".format(str(X_val.shape))
print "Val labels shape: {0}".format(str(Y_val.shape))
print "Test data shape: {0}".format(str(X_test.shape))
print "Test labels shape: {0}".format(str(Y_test.shape))


# DONT CHANGE THE SEED AND THE DEFAULT PARAMETERS. OTHERWISE WE WILL NOT BE ABLE TO CORRECT YOUR ASSIGNMENT!
# Seed
np.random.seed(42)

# Default parameters.
num_iterations = 1500
val_iteration = 100
batch_size = 200
learning_rate = 1e-7
weight_decay = 3e+4
weight_scale = 0.0001

########################################################################################
# TODO:                                                                                #
# Build the multinomial logistic regression classifier using the Network model. You    #
# will need to use add_layer and add_loss methods. Train this model using Solver class #
# with SGD optimizer. In configuration of the optimizer you need to specify only       #
# learning rate. Use the fit method to train classifier. Don't forget to include       #
# X_val and Y_val in arguments to output the validation loss and accuracy during       #
# training. Set the verbose to True to compare with the  multinomial logistic          #
# regression classifier from the Section 2.                                            #
########################################################################################
model = Network()
model.add_layer(LinearLayer(layer_params = {'input_size': X_train.shape[1], 'output_size':num_classes, 'weight_decay': weight_decay, 'weight_scale':weight_scale}))
model.add_layer(SoftMaxLayer(layer_params = {'input_size': num_classes, 'output_size':num_classes, 'weight_decay': weight_decay, 'weight_scale':weight_scale}))
model.add_loss(SoftMaxLoss)
optimizer = SGD()
optimizer_config = None
solver = Solver(model)
solver.fit(X_train, Y_train, optimizer, optimizer_config, X_val, Y_val, batch_size, num_iterations, val_iteration, verbose=True)
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################

########################################################################################
# TODO:                                                                                #
# Compute the accuracy on the test set.                                                #
########################################################################################
test_acc = None
########################################################################################
#                              END OF YOUR CODE                                        #
########################################################################################
print("Test Accuracy = {0:.3f}".format(test_acc))


