"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *

def hello_convolutional_networks():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from convolutional_networks.py!')


class Conv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input: 4차원 다룸!
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys: # conv_param에서 불러올 수 있는 값들
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ##############################################################################
    # TODO: Implement the convolutional forward pass.                            #
    # Hint: you can use the function torch.nn.functional.pad for padding.(함수 사용가능) #
    # Note that you are NOT allowed to use anything in torch.nn in other places. #
    ##############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape # 우선 x의 shape으로 값 가져오기
    F, C, HH, WW = w.shape 
    stride, pad = conv_param['stride'], conv_param['pad']
    
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride
    out = torch.zeros((N, F, H_new, W_new), device=x.device, dtype=x.dtype) # initializing output + x와 device, type을 맞춰줌.
    ## 유의사항
    # torch.nn.functional.pad(input, pad, mode='constant', value=0.0) 을 사용할 때 pad argument 부분을 유의하자!
    # to pad the last 2 dimensions of the input tensor, then use (padding_left, padding_right, padding_top, padding_bottom)
    # W와 H부분에 zero padding하기 (사각형 -> 4부분)
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad)).to(x.dtype).to(x.device) # ?언제 이런 to를 사용하는거지?
    
    # ? 이렇게 하면 위에서 정한 out shape과 맞나?
    for i in range(N):
        for j in range(F):
            for hh in range(0, H, stride):
                for ww in range(0, W, stride):
                    out[i, f, hh//stride, ww//stride] = (W[f]*x_pad[i,:,hh:hh+HH, ww:ww+WW]).sum() + b[f]
                                    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # Replace "pass" statement with your code
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    # Padding
    x_pad = torch.nn.functional.pad(x, (pad,pad,pad,pad)).to(x.dtype).to(x.device)
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride
    # Construct output
    dx_pad = torch.zeros_like(x_pad).to(x.dtype).to(x.device) # x_pad의 shape만큼 zero로
    dx, dw, db = [torch.zeros_like(i).to(i.dtype) for i in [x, w, b]] # x, w, b의 shape만큼 zero로 (리스트 안에 넣어 하나씩, 한번에 바꿔줌)
    
    for i in range(N):
        for j in range(F):
            db[f] += torch.sum(dout[i,f])
            for hh in range(0, H, stride):
                for ww in range(0, W, stride):
                    dw[f] += x_pad[i, :, hh:hh+HH, ww:ww+WW] * dout[i,f,hh//stride, ww//stride] # ??
                    dx_pad[i, :, hh:hh+HH, ww:ww+WW] += w[f] * dout[i,f,hh//stride, ww//stride]
                    
    dx = dx_pad[:,:, pad:pad+H, pad:pad+W] # ?
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max-pooling forward pass                              #
    #############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'],pool_param['pool_width'], pool_param['stride'] # filter size, stride 
    H_new = 1 + (H - pool_height) // stride
    W_new = 1 + (W - pool_width) // stride
    out = torch.zeros((N, C, H_new, W_new)).to(x.device).to(x.dtype)
    for hh in range(0, H, stride):
        for ww in range(0, W, stride):
            # hh와 ww로 이루어진 사각형에서 가장 큰 값 -> torch.max를 두번 사용해 2차원 tensor에서 가장 큰 값을 고른다.
            # torch.max를 사용하면 value만 나오지 않으므로 .values를 통해 값만 뽑아낸다. 
            out[:,:,hh//stride,ww//stride] = torch.max(torch.max(x[:,:,hh:hh+pool_height, ww:ww+pool_width], dim=2).values, dim=2).values
                          
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max-pooling backward pass                             #
    #############################################################################
    # Replace "pass" statement with your code
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param.get('pool_height', 2) # 딕셔너리에서 값 뽑아내는 다른 방법
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    H_new = 1 + (H - pool_height) // stride
    W_new = 1 + (W - pool_width) // stride
    # Construct output
    dx = torch.zeros_like(x).to(x.device)
    # Loops
    # max의 미분은 가장 큰 값이었던 부분만 살리고 나머지 0으로
    for i in range(N):
        for c in range(C):
            for hh in range(0, H, stride):
                for ww in range(0, W, stride):
                    # argmax는 tensor를 일차원으로 쭉 펴서 가장 높은 값을 가진 index 반환
                    ind = torch.argmax(x[i, c, hh:hh+pool_height, ww:ww+pool_width]) # 범위가 hh와 ww에만 있음! i와 c는 고정
                    ind1, ind2 = ind//pool_height, ind % pool_width # ?이렇게 하면 된다고?
                    dx[i, c, hh:hh+poo_height, ww:ww+pool_width][ind1,ind2] = dout[i,c, hh//stride, ww//stride]
                    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx # dx만 return 한다.


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=torch.float, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights. (weight에 곱할 표준편차) -> 왜 있는거야?
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian centered at 0.0   #
    # with standard deviation equal to weight_scale; biases should be          #
    # initialized to zero. All weights and biases should be stored in the      #
    #  dictionary self.params. Store weights and biases for the convolutional  #
    # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
    # weights and biases of the hidden linear layer, and keys 'W3' and 'b3'    #
    # for the weights and biases of the output linear layer.                   #
    #                                                                          #
    # IMPORTANT: For this assignment, you can assume that the padding          #
    # and stride of the first convolutional layer are chosen so that           #
    # **the width and height of the input are preserved**. Take a look at      #
    # the start of the loss() function to see how that happens.                #               
    ############################################################################
    # Replace "pass" statement with your code
    '''
    weights와 biases 초기화, self.params(딕셔너리 형태)에 저장하기.
    '''
    C, H, W = input_dims
    # 지금까지의 W와 shape같음!
    self.params['W1'] = weight_scale * torch.randn(num_filters, C, filter_size, filter_size).to(dtype).to(device)
    self.params['b1'] = torch.zeros(num_filters).to(dtype).to(device) # b1은 사전정보 없으므로 0으로 시작
    
    self.params['W2'] = weight_scale * torch.randn(num_filters * H * W // 4, hiddne_dim).to(dtype).to(device) # 이건 왜지..?왜 2차원?
    self.params['b2'] = torch.zeros(hidden_dim).to(dtype).to(device)
    
    self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes).to(dtype).to(device)
    self.params['b3'] = torch.zeros(num_classes).to(dtype).to(device)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # Remember you can use the functions defined in your implementation above. #
    ############################################################################
    # Replace "pass" statement with your code
    out1, cache1 = Conv_ReLU_Pool.forward(
        X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    out2, cache2 = Linear_ReLU.forward(
        out1, self.params['W2'], self.params['b2'])
    out3, cache3 = Linear.forward(
        out2, self.params['W3'], self.params['b3'])
    scores = out3
                                       
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    reg = torch.tensor(self.reg) # reg를 tensor로 - L2 regularization
    loss, dout = softmax_loss(out3, y)
    loss += reg * torch.sum(torch.tensor([torch.sum(self.params['W%d'%i] ** 2) for i in [1,2,3]]))
    # 뒤에서부터
    dout, grads['W3'], grads['b3'] = Linear.backward(dout, cache3)
    grads['W3'] += 2 * reg * self.params['W3'] # L2 regularization 고려하기 (미분한 형태)
    dout, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout, cache2)
    grads['W2'] += 2 * reg * self.params['W2']
    _, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout, cache1) # dout 더이상 안 쓰므로 받아주지 않음.
    grads['W1'] += 2 * reg * self.params['W1']
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8], # 필터씌우기가 5번 있는거?
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively. = 감마는 1로, 베타는 0으로 초기화          #           
    ############################################################################
    # Replace "pass" statement with your code
    layers_dims = num_filters + [num_classes] # num_filters 리스트에 합치기
    C, H, W = input_dims
    
    # 왜 num_filters를 기준으로 하는 거?
    
    for i in range(len(num_filters)):
        if i == 0:
            if weight_scale == 'kaiming':
                # params딕셔너리에 key와 value형태로 저장됨 / 앞에 f를 써주면 i에 지정한 값을 넣어서 {i+1}에서 계산된 숫자 사용
                self.params[f'W{i+1}'] = kaiming_initializer(
                    layers_dims[i], C, K=3, relu=True, device=device, dtype=dtype)
            else:
                self.params[f'W{i+1}'] = weight_scale * torch.randn(layers_dims[i], C, 3,3).to(dtype).to(device)
            
            self.params[f'b{i+1}'] = torch.zeros(layers_dims[i]).to(dtype).to(device)
        else:
                if weight_scale == "kaiming":
                    self.params[f'W{i+1}'] = kaiming_initializer(layers_dims[i], layers_dims[i-1],  K=3,
                                                                 relu=True, device=device, dtype=dtype)
                else:
                    self.params[f'W{i+1}'] = weight_scale * torch.randn(
                        layers_dims[i], layers_dims[i-1], 3, 3).to(dtype).to(device)
                
                self.params[f'b{i+1}'] = torch.zeros(layers_dims[i]).to(dtype).to(device)
                
        if self.batchnorm == True: # Whether로 고르라고 함
            for i in range(len(num_filters)):
                self.params[f'gamma{i+1}'] = torch.ones(layers_dims[i]).to(dtype).to(device)
                self.params[f'beta{i+1}'] = torch.zeros(layers_dims[i]).to(dtype).to(device)
        
        
        # downsample은 왜 하는거지?
        i += 1
        downsample_rate = 2**(len(max_pools))
        # print("downsample_rate:", downsample_rate)
        if weight_scale == "kaiming":
            self.params[f'W{i+1}'] = kaiming_initializer(num_filters[i-1] * H * W // downsample_rate**2,
                                                         layers_dims[i], K=None,
                                                         relu=True, device=device, dtype=dtype)
        else:
            self.params[f'W{i+1}'] = weight_scale * torch.randn(num_filters[i-1]
                                                                * H * W // downsample_rate**2, layers_dims[i]).to(dtype).to(device)
        # print("Linear weight shape:", self.params['W'+str(i+1)].shape)
        self.params[f'b{i+1}'] = torch.zeros(layers_dims[i]
                                             ).to(dtype).to(device)
        
                
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################
    # Replace "pass" statement with your code
    reg = torch.tensor(self.reg).to(X.dtype).to(X.device)
    scores = X
    cache_history = []
    L2reg = 0
    
    for i in range(self.num_layers-1): # range(5)
        scores,cache = FastConv.forward(scores, self.params[f'W{i+1}'],
                                        self.params[f'b{i+1}'], conv_param)
        cache_history.append(cache)
        L2reg += torch.sum(self.params[f'W{i+1}'] ** 2) # L2 regularization
        
        if self.batchnorm: # 만약 batchnorm을 한다면
            scores, cache = SpatialBatchNorm.forward(scores, self.params[f'gamma{i+1}'],
                                                         self.params[f'beta{i+1}'],
                                                         self.bn_params[i])
            cache_history.append(cache)
        
        scores, cache = ReLU.forward(scores)
        cache_history.append(cache)
        
        if i in self.max_pools: # 만약 max_pooling을 한다면
            scores,cache = FastMaxPool.forward(scores, pool_param)
            cache_history.append(cache)
            
        i += 1 # 왜 여기서 +1을 하는거지?
        scores, cache = Linear.forward(
            scores, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
        
        cache_history.append(cache)
        L2reg += torch.sum(self.params[f'W{i+1}'] ** 2)
        L2reg *= self.reg
            
            
            
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss, dout = softmax_loss(scores, y)
    loss += L2reg
    
    dout, grads[f'W{i+1}'], grads[f'b{i+1}'] = Linear.backward(dout, cache_history.pop())
    
    grads[f'W{i+1}'] += 2 * reg * self.params[f'W{i+1}'] # L2 regularization 미분
    
    i -= 1 # i=5로 끝나서 -1하면 4가 됨.
    while i >= 0:
        if i in self.max_pools:
            dout = FastMaxPool.backward(dout, cache_history.pop())
        dout = ReLU.backward(dout, cache_history.pop())
        if self.batchnorm:
            dout, grads[f'gamma{i+1}'], grads[f'beta{i+1}'] = SpatialBatchNorm.backward(dout, cache_history.pop())
        dout, grads[f'W{i+1}'], grads[f'b{i+1}'] = FastConv.backward(dout, cache_history.pop())
        grads[f'W{i+1}'] += 2 * reg * self.params[f'W{i+1}']
        i -= 1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def find_overfit_parameters():
  weight_scale = 2e-3   # Experiment with this!
  learning_rate = 1e-5  # Experiment with this!
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 30 epochs.                                      #
  ############################################################################
  # Replace "pass" statement with your code
  pass
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
  ################################################################################
  # TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
  ################################################################################
  # Replace "pass" statement with your code
    data = {
        'X_train' : data_dict['X_train'],
        'y_train' : data_dict['y_train'],
        'X_val' : data_dict['X_val'],
        'y_val' : data_dict['y_val']
    }
    num_epochs = 50 #왜?
    lr = 1e-3
    
    model = DeepConvNet(input_dims = input_dims, num_classes=10,
                        num_filters = [32,32,64,64,128,128],
                        max_pools = [1,3,5],
                        weight_scale='kaiming',
                        batchnorm=False,
                        reg=1e-5, device='cuda')
    
    # Solver는 import 해서 쓰는 것
    solver = Solver(model, data, num_epochs = num_epochs, batch_size=128,
                    update_rule=adam, optim_config={'learning_rate':lr},
                    lr_decay=0.9,
                    verbose=False, device='cuda')
    

      
  ################################################################################
  #                              END OF YOUR CODE                                #
  ################################################################################
  return solver

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
    shape은 주어졌고, 값은 tensor로 랜덤으로 받으면 된다. 
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    if relu:
        weight = (2/Din) ** (1/2) * torch.randn(Din, Dout, dtype=dtype, device=device)
        
    else:
        weight = (1/Din) ** (1/2) * torch.randn(Din, Dout, dtype=dtype, device=device)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    if relu:
            weight = (2/(Din*K*K))**(1/2) * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
                                                        
    else:
        weight = (1/(Din*K*K))**(1/2) * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)
                                                    

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5) # eps라는 key가 없으면 1e-5를 반환해라
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      # Replace "pass" statement with your code
        mu = torch.mean(x,0) # 주로 각 feature별 평균
        var = 1. / N * torch.sum((x-mu) ** 2, 0)
        x_hat = (x - mu) / ((var + eps)** (1/2)) # eps는 뭐야?
        y = gamma * x_hat + beta  # 최종 y를 낼 때는 gamma와 beta 사용해야 함! <- 0부근에서 선형구간으로 빠지는 문제 해결위해
        out = y
        running_mean = momentum * running_mean + (1 - momentum) * mu # 테스트에 사용?
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = (x_hat, mu, var, eps, gamma, beta, x)
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      # Replace "pass" statement with your code
      x_hat = (x - running_mean) / sqrt((running_var + eps))
      y = gamma * x_hat + beta
      out = y  
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code
    ## 이해안됨.. ##
    x_hat, mu, var, eps, gamma, beta, x = cache
    N, D = dout.shape

    dbeta = torch.sum(dout, 0)
    dgamma = torch.sum(dout * x_hat, 0)

    dx_hat = dout * gamma # 여기도 위와 같은 방식으로 이상한데..?
    dxmu1 = dx_hat * 1 / (var + eps)**(1/2)
    divar = torch.sum(dx_hat * (x - mu), 0)
    dvar = divar * -1 / 2 * (var + eps) ** (-3/2)
    dsq = 1 / N * torch.ones((N, D)).to(x.dtype).to(x.device) * dvar
    dxmu2 = 2 * (x - mu) * dsq
    dx1 = dxmu1 + dxmu2
    dmu = -1 * torch.sum(dxmu1 + dxmu2, 0)
    dx2 = 1 / N * torch.ones((N, D)).to(x.dtype).to(x.device) * dmu
    dx = dx1 + dx2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Replace "pass" statement with your code
    ## 이해안됨 ## 
    (x_hat, mu, var, eps, gamma, beta, x), N = cache, dout.shape[0]
    dbeta = dout.sum(0)
    dgamma = (dout * x_hat).sum(0)
    dx = (dout * gamma - (dout * gamma).sum(0)/N - (dout * gamma * x_hat).sum(0) * x_hat/N)/var**(1/2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape
    x = x.permute(0, 2, 3, 1).reshape(N*H*W, C) # 왜 이렇게 변형해서 사용하는거지?? (모르겠음)
    out, cache = BatchNorm.forward(x, gamma, beta, bn_param)
    out = out.reshape(N,H,W,C).permute(0,3,1,2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    N, C, H, W = dout.shape
    dout = dout.permute(0, 2, 3, 1).reshape(N*H*W, C)
    dx, dgamma, dbeta = BatchNorm.backward_alt(dout, cache)
    dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


################################################################################
################################################################################
#################   Fast Implementations and Sandwich Layers  ##################
################################################################################
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db

class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta
