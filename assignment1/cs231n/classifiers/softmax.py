import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    scores = np.exp(scores) / np.sum(np.exp(scores))
    correct_class_score = scores[y[i]]
    loss -= np.log(correct_class_score)
    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += X[i] * (correct_class_score - 1)
      else:
        dW[:, j] += X[i] * scores[j]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, -1)
  score_y = scores[np.arange(num_train), y]
  loss = np.sum(-np.log(score_y)) / num_train
  loss += reg * np.sum(W * W)

  scores[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, scores) / num_train
  dW += reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

