import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,32], filter_size=[7,7],
               hidden_dim=[512,128], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, dropout=0.0,seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_filters = num_filters
    self.hidden_dim = hidden_dim
    self.filter_size = filter_size
    self.use_dropout = dropout > 0
    
    stride = 1
    pad = (filter_size[0] - 1) / 2
    
    depth, width, height = input_dim

    for i in range(0,len(num_filters)):
      layer_size = [num_filters[i], depth, filter_size[i], filter_size[i]]
      layer_num = layer_size[0] * layer_size[1] * layer_size[2] * layer_size[3]
      
      self.params['theta'+str(i+1)] = np.reshape(np.random.normal(0, weight_scale, layer_num), layer_size)
      self.params['theta'+str(i+1)+'_0'] = np.zeros(shape=[num_filters[i]])
    
      depth = num_filters[i]
      width = 1 + (width + 2 * pad - filter_size[i]) / stride
      height = 1 + (height + 2 * pad - filter_size[i]) / stride
    
    
    conv_out_num = depth * width * height / 4
    dims = [conv_out_num,]*len(num_filters) + hidden_dim + [num_classes]
    
    for i in range(len(num_filters), len(num_filters)+len(hidden_dim) + 1):
      self.params['theta'+str(i+1)] = np.reshape(np.random.normal(0, weight_scale, dims[i-1]*dims[i]),[dims[i-1], dims[i]])
      
      self.params['theta'+str(i+1)+'_0'] = np.zeros(shape=[dims[i]])


    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
   
    for k, v in self.params.iteritems():
      print k, np.shape(v)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    """
    mode = 'test' if y is None else 'train'

    # Set train/test mode for  dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
   
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.filter_size[0]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # about 3 lines of code (use the helper functions in layer_utils.py)
    cache_dict = {}
    drop_cache_dict = {}
    out = X

    # adds the conv_relu layers
    for i in range(0, len(self.num_filters)-1):
      out, cache = conv_relu_forward(out,
                                     self.params['theta'+str(i+1)],
                                     self.params['theta'+str(i+1)+'_0'],
                                     conv_param)
      cache_dict['theta'+str(i+1)] = cache
    
    # The last conv layer has a max pool 
    out, cache = conv_relu_pool_forward(out,
                                        self.params['theta'+str(len(self.num_filters))],
                                        self.params['theta'+str(len(self.num_filters))+'_0'],
                                        conv_param, pool_param)
    cache_dict['theta'+str(len(self.num_filters))] = cache
    
    #Adds the fully connected layers
    for i in range(len(self.num_filters), len(self.num_filters)+len(self.hidden_dim)):
      out, cache = affine_relu_forward(out,
                                       self.params['theta'+str(i+1)],
                                       self.params['theta'+str(i+1)+'_0'])
      cache_dict['theta'+str(i+1)] = cache
      if self.use_dropout:
        out, drop_cache = dropout_forward(out, self.dropout_param)
        drop_cache_dict['theta'+str(i+1)] = drop_cache

    scores, cache = affine_forward(out,
                                   self.params['theta'+str(1+len(self.num_filters)+len(self.hidden_dim))],
                                   self.params['theta'+str(1+len(self.num_filters)+len(self.hidden_dim))+'_0'])
                                          
    cache_dict['theta'+str(1+len(self.num_filters)+len(self.hidden_dim))] = cache

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # about 12 lines of code
    i = len(self.num_filters)+len(self.hidden_dim)

    loss, dx = softmax_loss(scores, y)
    dx, grads['theta'+str(i+1)], grads['theta'+str(i+1)+'_0'] = affine_backward(dx,
                                                                                cache_dict['theta'+str(i+1)])

    for i in range(len(self.num_filters)+len(self.hidden_dim)-1, len(self.num_filters)-1, -1):
      if self.use_dropout:
        dx = dropout_backward(dx, drop_cache_dict['theta'+str(i+1)])

      dx, grads['theta'+str(i+1)], grads['theta'+str(i+1)+'_0'] = affine_relu_backward(dx,
                                                                                       cache_dict['theta'+str(i+1)])
    i = len(self.num_filters)
    dx, grads['theta'+str(i)], grads['theta'+str(i)+'_0'] = conv_relu_pool_backward(dx,
                                                                               cache_dict['theta'+str(i)])
    for i in range(len(self.num_filters)-1, 0, -1):
      dx, grads['theta'+str(i)], grads['theta'+str(i)+'_0'] = conv_relu_backward(dx,
                                                                            cache_dict['theta'+str(i)])
    for i in range(1,2+len(self.num_filters)+len(self.hidden_dim)):
      loss += 0.5 * self.reg * np.sum(np.square(self.params['theta'+str(i)]))
      grads['theta'+str(i)] += self.reg * self.params['theta'+str(i)]

    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

