import numpy as np
"""
This module implements various losses for the network.
You should fill in code into indicated sections.
"""

def HingeLoss(x, y):
  """
  Computes multi-class hinge loss and gradient of the loss with the respect to the input for multiclass SVM.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar hinge loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute hinge loss on input x and y and store it in loss variable. Compute gradient  #
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx

def CrossEntropyLoss(x, y):
  """
  Computes multi-class cross entropy loss and gradient with the respect to the input x.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar multi-class cross entropy loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute cross entropy loss on input x and y and store it in loss variable. Compute   #
  # gradient of the loss with respect to the input and store it in dx.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx


def SoftMaxLoss(x, y):
  """
  Computes the loss and gradient with the respect to the input x.

  Args:
    x: Input data.
    y: Labels of data.

  Returns:
    loss: Scalar softmax loss.
    dx: Gradient of the loss with the respect to the input x.

  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute softmax loss on input x and y and store it in loss variable. Compute gradient#
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  exp_nor = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
  loss = -np.sum(np.log(exp_nor[range(y.shape[0]), y])) / y.shape[0]

  dx = exp_nor
  dx[range(y.shape[0]), y] -= 1
  dx /= y.shape[0]
  # print "Error SoftMaxLoss", loss, dx.shape
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################
  return loss, dx
