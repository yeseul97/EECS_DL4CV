"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod

def hello_linear_classifier():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from linear_classifier.py!')


# Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier(object):
  """ An abstarct class for the linear classifiers """
  # Note: We will re-use `LinearClassifier' in both SVM and Softmax
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                  num_iters, batch_size, verbose)
    self.W, loss_history = train_linear_classifier(*train_args)
    return loss_history

  def predict(self, X):
    return predict_linear_classifier(self.W, X)

  @abstractmethod
  def loss(self, W, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative.
    Subclasses will override this.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
    - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an tensor of the same shape as W
    """
    raise NotImplementedError

  def _loss(self, X_batch, y_batch, reg):
    self.loss(self.W, X_batch, y_batch, reg)

  def save(self, path):
    torch.save({'W': self.W}, path)
    print("Saved in {}".format(path))

  def load(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("load checkpoint file: {}".format(path))



class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """
  def loss(self, W, X_batch, y_batch, reg):
    return softmax_loss_vectorized(W, X_batch, y_batch, reg)



#**************************************************#
################## Section 1: SVM ##################
#**************************************************#

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples. When you implment the regularization over W, please DO NOT
  multiply the regularization term by 1/2 (no coefficient).

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  dW = torch.zeros_like(W) # initialize the gradient as zero

 # compute the loss and the gradient
  num_classes = W.shape[1] # 클래스 개수 C
  num_train = X.shape[0] # 한 배치의 데이터 개수 N
  loss = 0.0
  for i in range(num_train):
    scores = W.t().mv(X[i]) # torch.mv( W.t(), X[0]) 과 같음(행렬곱) 3*5과 5*1 = 8*1 (클래스별로 weight곱한 output값 나옴)
    correct_class_score = scores[y[i]] # 각 데이터에서 class 개수만큼의 output이 나오고, 맞는 클래스에 해당하는 score만 저장
    for j in range(num_classes):
      if j == y[i]:
        continue # 아래 코드를 실행하지 않고 건너뜀
      margin = scores[j] - correct_class_score + 1 # note delta = 1 (강의에 나온 공식과 같은 형태)
      if margin > 0: # max(0,값) 이기 때문에 0을 넘는 값만 취급
        loss += margin # 다른 클래스로 예측한 score 값들과 실제 score값의 차이는 loss로 판단

        #######################################################################
        # TODO:                                                               #
        # Compute the gradient of the loss function and store it dW. (part 1) #
        # Rather than first computing the loss and then computing the         #
        # derivative, it is simple to compute the derivative at the same time #
        # that the loss is being computed.                                    #
        #######################################################################
        # Replace "pass" statement with your code
        dW[:,y[i]] += -X[i].T
        dW[:,j] += X[i].T # j클래스의 미분값 = input값을 다 더한것과 같음 (Score = WX -> W에 대해 미분하면 X만 남음)
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * torch.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it in dW. (part 2)    #
  #############################################################################
  # Replace "pass" statement with your code
  dW /= num_train
  dW += 2*reg*W # L2 = reg x W^2 -> L2 규제를 W에 대해 미분한 것
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation. When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as torch scalar
  - gradient of loss with respect to weights W; a tensor of same shape as W
  """
  loss = 0.0
  dW = torch.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_classes = W.shape[1] # 클래스 개수 C
  num_train = X.shape[0] # 한 배치의 데이터 개수 N
  scores = X.mm(W)
  correct_class_scores = scores[range(num_train), y].reshape(-1, 1) 
  # range로 데이터 개수만큼 돌면서, 본인 클래스에 해당하는 score값 저장
  margin = torch.maximum(torch.zeros((num_classes)),
                         scores - correct_class_scores +1)
  margin[range(num_train),y]=0 # 본인 클래스에 해당하는 margin값은 0으로 바꿈.
  loss = margin.sum() / num_train 
  loss += reg * torch.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Replace "pass" statement with your code
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis=1) # 데이터별로(행별로) margin이 존재하는 개수
  margin[torch.arange(num_train,y)]  -= valid_margin_count # 위에서 구한 개수에 마이너스를 붙이기
  dW = (X.T).mm(margin) / num_train # X에 위에서 구한 음수값 곱하기
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW


def sample_batch(X, y, num_train, batch_size):
  """
  Sample batch_size elements from the training data and their
  corresponding labels to use in this round of gradient descent.
  """
  X_batch = None
  y_batch = None
  #########################################################################
  # TODO: Store the data in X_batch and their corresponding labels in     #
  # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
  # and y_batch should have shape (batch_size,)                           #
  #                                                                       #
  # Hint: Use torch.randint to generate indices.                          #
  #########################################################################
  # Replace "pass" statement with your code
  batch_idxes = torch.randint(num_train,) # randint(범위,뽑을 크기)
  X_batch = X[batch_idxes]
  y_batch = y[batch_idxes]
  #########################################################################
  #                       END OF YOUR CODE                                #
  #########################################################################
  return X_batch, y_batch


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200,
                            verbose=False):
  """
  Train this linear classifier using stochastic gradient descent.

  Inputs:
  - loss_func: loss function to use when training. It should take W, X, y
    and reg as input, and output a tuple of (loss, dW)
  - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
    classifier. If W is None then it will be initialized here.
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
    means that X[i] has label 0 <= c < C for C classes.
  - learning_rate: (float) learning rate for optimization.
  - reg: (float) regularization strength.
  - num_iters: (integer) number of steps to take when optimizing
  - batch_size: (integer) number of training examples to use at each step.
  - verbose: (boolean) If true, print progress during optimization.

  Returns: A tuple of:
  - W: The final value of the weight matrix and the end of optimization
  - loss_history: A list of Python scalars giving the values of the loss at each
    training iteration.
  """
  # assume y takes values 0...K-1 where K is number of classes
  num_train, dim = X.shape
  if W is None:
    # lazily initialize W
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  # Run stochastic gradient descent to optimize W
  loss_history = []
  for it in range(num_iters):
    # TODO: implement sample_batch function
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # evaluate loss and gradient
    loss, grad = loss_func(W, X_batch, y_batch, reg)
    loss_history.append(loss.item())

    # perform parameter update
    #########################################################################
    # TODO:                                                                 #
    # Update the weights using the gradient and the learning rate.          #
    #########################################################################
    # Replace "pass" statement with your code
    W -= learning_rate*grad # weight 업데이트 하는 공식
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss))

  return W, loss_history


def predict_linear_classifier(W, X):
  """
  Use the trained weights of this linear classifier to predict labels for
  data points.

  Inputs:
  - W: A PyTorch tensor of shape (D, C), containing weights of a model
  - X: A PyTorch tensor of shape (N, D) containing training data; there are N
    training samples each of dimension D.

  Returns:
  - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
    elemment of X. Each element of y_pred should be between 0 and C - 1.
  """
  y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
  ###########################################################################
  # TODO:                                                                   #
  # Implement this method. Store the predicted labels in y_pred.            #
  ###########################################################################
  # Replace "pass" statement with your code
  class_preds = X.mm(W)
  y_pred = torch.argmax(class_preds,axis=1) # output에서 데이터별로 가장 높은 값을 가지는 클래스 선택(클래스:0부터C-1)
  # torch 함수들 공부 필요할듯
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################
  return y_pred


def svm_get_search_params():
  """
  Return candidate hyperparameters for the SVM model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """

  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  # TODO:   add your own hyper parameter lists.                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rate = [0.1,0.01,0.001]
  regularization_strengths = [1,0.1,0.01]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths


def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
  """
  Train a single LinearClassifier instance and return the learned instance
  with train/val accuracy.

  Inputs:
  - cls (LinearClassifier): a newly-created LinearClassifier instance.
                            Train/Validation should perform over this instance
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - lr (float): learning rate parameter for training a SVM instance.
  - reg (float): a regularization weight for training a SVM instance.
  - num_iters (int, optional): a number of iterations to train

  Returns:
  - cls (LinearClassifier): a trained LinearClassifier instances with
                            (['X_train', 'y_train'], lr, reg)
                            for num_iter times.
  - train_acc (float): training accuracy of the svm_model
  - val_acc (float): validation accuracy of the svm_model
  """
  train_acc = 0.0 # The accuracy is simply the fraction of data points
  val_acc = 0.0   # that are correctly classified.
  ###########################################################################
  # TODO:                                                                   #
  # Write code that, train a linear SVM on the training set, compute its    #
  # accuracy on the training and validation sets                            #
  #                                                                         #
  # Hint: Once you are confident that your validation code works, you       #
  # should rerun the validation code with the final value for num_iters.    #
  # Before that, please test with small num_iters first                     #
  ###########################################################################
  # Feel free to uncomment this, at the very beginning,
  # and don't forget to remove this line before submitting your final version
  # num_iters = 100

  # Replace "pass" statement with your code
  X_train, y_train = data_dict['X_train'],data_dict['y_train']
  X_val, y_val = data_dict['X_val'], data_dict['y_val']
  loss = cls.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=100)

  y_train_pred = cls.predict(X_train)
  train_acc = (y_train_pred == y_train).sum()/len(y_train)
  y_val_pred = cls.predict(X_val)
  val_acc = (y_val_pred == y_val).sum()/len(y_val)
  ############################################################################
  #                            END OF YOUR CODE                              #
  ############################################################################

  return cls, train_acc, val_acc



#**************************************************#
################ Section 2: Softmax ################
#**************************************************#

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops).  When you implment
  the regularization over W, please DO NOT multiply the regularization term by
  1/2 (no coefficient).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A PyTorch tensor of shape (D, C) containing weights.
  - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
  - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an tensor of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  scores = X.mm(W)
  maximum_score = torch.max(scores,1)[0].reshape(-1,1) # 행에서 가장 큰 값
  scores -= maximum_score # 왜 빼는거?

  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_datapt_dims = X.shape[1] # 피쳐 개수

  for i in range(num_train):
      denom = 0.0
      for j in range(num_classes):
          denom += torch.exp(scores[i,j]) # 한 데이터에 대해서 모든 클래스의 점수 구해서 더함
      l_i = -1 * scores[i,y[i]] + torch.log(denom) # 공식임.
      loss += l_i
      
      for j in range(num_classes):
          dW[:,j] += (1/denom) * torch.exp(scores[i,j]) * X[i] # 식의 뒷부분 미분식
          if j == y[i]:
              dW[:,y[i]] -= X[i]

  loss /= num_train 
  dW /= num_train
  
  # add the regularization
  loss += reg * torch.sum(W*W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.  When you implment the
  regularization over W, please DO NOT multiply the regularization term by 1/2
  (no coefficient).

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = torch.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability (Check Numeric Stability #
  # in http://cs231n.github.io/linear-classify/). Don't forget the            #
  # regularization!                                                           #
  #############################################################################
  # Replace "pass" statement with your code
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  scores = X.mm(W)

  maximum_score = torch.max(scores, 1)[0].reshape(-1, 1)
  scores -= maximum_score

  expscores = torch.exp(scores)

  denoms = torch.sum(expscores,1) # 행별로 더하기 
  numers = expscores[range(num_train),y]
  loss = -1 * torch.log(numers/denoms).sum()

  # Now compute dW
  correct_class = torch.zeros((num_train, num_classes), dtype=  X.dtype) # (데이터+클래스)영행렬 만들기
  correct_class[range(num_train),y] = -1 # 클래스에 모두 -1 넣기
  dW_term1 = X.t().mm(correct_class)
  dW_term2 = (X.t() / denoms).mm(expscores)
  dW = dW_term1 + dW_term2

  loss /= num_train
  dW /= num_train

  # add the regularization
  loss += reg * torch.sum(W*W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_get_search_params():
  """
  Return candidate hyperparameters for the Softmax model. You should provide
  at least two param for each, and total grid search combinations
  should be less than 25.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  """
  learning_rates = []
  regularization_strengths = []

  ###########################################################################
  # TODO: Add your own hyper parameter lists. This should be similar to the #
  # hyperparameters that you used for the SVM, but you may need to select   #
  # different hyperparameters to achieve good performance with the softmax  #
  # classifier.                                                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
  regularization_strengths = [1e-1, 1e-2, 1e-3]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, regularization_strengths
