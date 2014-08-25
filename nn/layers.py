# -*- coding: utf-8 -*-

"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""

import logging
#from util import *
import numpy as np
import numpy

import theano.tensor as T
from theano.tensor.nnet import conv
import theano

from interfaces import Layer

logger=logging.getLogger("layers")

class EmbeddingLayer(Layer):
  """
  EmbeddingsLayer is a layer where a lookup operation is performed.
  Indices supplied as input replaced with their embedding representation.
  """
  def __init__(self, input, w_values,embedding_dim,mini_batch_size,width, name="EmbeddingLayer", params=None):
    self.n_in = 1.0
    self.w_values=w_values
    super(EmbeddingLayer, self).__init__(name,params)
    concatenated_input = input.flatten()
    if theano.config.device == 'cpu':
        indexed_rows = theano.sparse_grad(self.weights[concatenated_input])
    else:
        indexed_rows = self.weights[concatenated_input]
    concatenated_rows = indexed_rows.flatten()
    #mini_batch_size = input.shape[0]
    #width = input.shape[1]
    self.output = concatenated_rows.reshape((mini_batch_size, width*embedding_dim))
    self.params = [self.weights]
    
  def initParams(self):
    self.weights = theano.shared(np.asarray(self.w_values, dtype=theano.config.floatX), name='weights_'+self.name, borrow=True)

class SharedEmbeddingLayer(Layer):
  """
  SharedEmbeddingsLayer is a layer where a lookup operation is performed.
  Indices supplied as input replaced with their embedding representation.
  The embeddings are given as a shared theano variable
  """
  def __init__(self, input, w_values,embedding_dim,mini_batch_size,width, name="SharedEmbeddingLayer"):
    self.name = name
    self.input = input
    self.n_in = 1.0
    self.weights=w_values
#    super(SharedEmbeddingLayer, self).__init__(name)
    concatenated_input = input.flatten()
    if theano.config.device == 'cpu':
        indexed_rows = theano.sparse_grad(self.weights[concatenated_input])
    else:
        indexed_rows = self.weights[concatenated_input]
    concatenated_rows = indexed_rows.flatten()
    #mini_batch_size = input.shape[0]
    #width = input.shape[1]
    self.output = concatenated_rows.reshape((mini_batch_size, width*embedding_dim))
    self.params = []
    
  def initParams(self):
    return


class StaticEmbeddingLayer(Layer):
  """
  StaticEmbeddingsLayer is a layer where a lookup operation is performed.
  Indices supplied as input replaced with their embedding representation.
  In contrast to EmbeddingLayer, the wordembeddings are not updated
  """
  def __init__(self, input, w_values,embedding_dim,mini_batch_size,width, name="StaticEmbeddingLayer"):
    self.name = name
    self.input = input
    self.n_in = 1.0
    self.n_out=width*embedding_dim
    self.w_values=w_values
    super(StaticEmbeddingLayer, self).__init__(name)
    concatenated_input = input.flatten()
    if theano.config.device == 'cpu':
        indexed_rows = theano.sparse_grad(self.weights[concatenated_input])
    else:
        indexed_rows = self.weights[concatenated_input]
    concatenated_rows = indexed_rows.flatten()
    #mini_batch_size = input.shape[0]
    #width = input.shape[1]
    self.output = concatenated_rows.reshape((mini_batch_size, width*embedding_dim))
    self.params = []
    
  def initParams(self):
    self.weights = theano.shared(np.asarray(self.w_values, dtype=theano.config.floatX), name='weights_'+self.name, borrow=True)

class StaticEmbeddingLayer3d(Layer):
  """
  StaticEmbeddingsLayer is a layer where a lookup operation is performed.
  Indices supplied as input replaced with their embedding representation.
  In contrast to EmbeddingLayer, the wordembeddings are not updated
  """
  def __init__(self, input, w_values,embedding_dim,mini_batch_size,width, name="StaticEmbeddingLayer"):
    self.name = name
    self.input = input
    self.n_in = 1.0
    self.n_out=width*embedding_dim
    self.w_values=w_values
    super(StaticEmbeddingLayer3d, self).__init__(name)
    concatenated_input = input.flatten()
    if theano.config.device == 'cpu':
        indexed_rows = theano.sparse_grad(self.weights[concatenated_input])
    else:
        indexed_rows = self.weights[concatenated_input]
    concatenated_rows = indexed_rows.flatten()
    #mini_batch_size = input.shape[0]
    #width = input.shape[1]
    self.output = concatenated_rows.reshape((mini_batch_size, width, embedding_dim))
    self.params = []
    
  def initParams(self):
    self.weights = theano.shared(np.asarray(self.w_values, dtype=theano.config.floatX), name='weights_'+self.name, borrow=True)

class ConvolutionLayer(Layer):
    """
    convolutional layer, using the tricks discussed in:
    https://groups.google.com/forum/#!searchin/theano-users/sander$20dieleman$20convolution/theano-users/JJHZmuUDdPE/ycnLznRePUgJ
    input:
        matrix with each row containing the embedding of a word
    output:
        output is a matrix with with each row containing hidden representation of word based on window ...
    """
    def __init__(self, input, batch_size, n_out, window_size, embedding_dim,activation, size_restriction, name="ConvolutionLayer",params=None):
        self.n_in=window_size*embedding_dim
        self.n_out=n_out
        self.window_size=window_size
        self.embedding_dim=embedding_dim
        self.activation=activation              
        super(ConvolutionLayer, self).__init__(name,params)    
        shuffled_input=input.dimshuffle(0, 'x', 2, 1)
        filter_shape = [n_out, 1, embedding_dim, window_size] 
        image_shape = [batch_size,1,embedding_dim,size_restriction] 
        tmp = conv.conv2d(shuffled_input, self.weights, border_mode='valid', image_shape=image_shape, filter_shape=filter_shape) 
        self.output = tmp.dimshuffle(0,3, 1) + self.b.dimshuffle('x','x', 0)
        self.L1 = abs(self.weights).sum()
        self.L2 = (self.weights ** 2).sum()
        self.params = [self.weights, self.b]

    def initParams(self):
        w_values = random_value((self.n_out, 1, self.embedding_dim, self.window_size), theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            w_values *= 4
        self.weights = theano.shared(value=w_values, name='weights_'+self.name, borrow=True)
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_'+self.name, borrow=True)

class MaxLayer(Layer):
    """
    MaxLayer described in SENNA-paper
    input:
        matrix where each column represents one dimension of different vectors
    output:
        vector-row containing the maximum value for each dimension
    """
    def __init__(self, input, name="MaxLayer"):
        self.name = name
        self.output = input.max(axis=1)
        self.n_in=1.0
        self.params=[]
class DoubleEmbeddingHiddenLayer(Layer):
    def __init__(self,  input, embeddings_values, features_values, n_in1, n_in2, batch_size, n_out, rng,activation=T.tanh,name="DoubleEmbeddingHiddenLayer",
                 params=None):
        self.n_in=[n_in1,n_in2]
        self.n_out=n_out
        self.activation=activation
        self.rng=rng
        self.w_values=embeddings_values
        self.f_values=features_values
        super(DoubleEmbeddingHiddenLayer, self).__init__(name,params)   
        lin_output = T.dot(self.embeddings[input[:,:,0].flatten()].reshape((batch_size,n_in1)), self.W1)+ T.dot(self.afeatures[input[:,:,1].flatten()].reshape((batch_size,n_in2)), self.W2)+ self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        self.L1 = abs(self.W1).sum()+abs(self.W2).sum()
        self.L2 = (self.W1 ** 2).sum() + (self.W2 ** 2).sum()
        self.params = [self.W1, self.W2, self.b,self.embeddings]
        self.n_in=n_in1+n_in2
    
    def initParams(self):
        W1_values = np.asarray(self.rng.uniform(
    #       low=-np.sqrt(6. / (n_in + n_out)),
    #       high=np.sqrt(6. / (n_in + n_out)),
            low=-np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
            high=np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
    #       low=-np.sqrt(3. / np.sqrt(n_in)),
    #       high=np.sqrt(3. / np.sqrt(n_in)),
        size=(self.n_in[0], self.n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W1_values *= 4
        self.W1 = theano.shared(value=W1_values, name='W1_'+self.name, borrow=True)
        
        W2_values = np.asarray(self.rng.uniform(
    #       low=-np.sqrt(6. / (n_in + n_out)),
    #       high=np.sqrt(6. / (n_in + n_out)),
            low=-np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
            high=np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
    #       low=-np.sqrt(3. / np.sqrt(n_in)),
    #       high=np.sqrt(3. / np.sqrt(n_in)),
        size=(self.n_in[1], self.n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W2_values *= 4
        self.W2 = theano.shared(value=W2_values, name='W2_'+self.name, borrow=True)
        
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_'+self.name, borrow=True)
        
        self.embeddings = theano.shared(np.asarray(self.w_values, dtype=theano.config.floatX), name='embeddings_'+self.name, borrow=True)
        self.afeatures = theano.shared(np.asarray(self.f_values, dtype=theano.config.floatX), name='afeatures_'+self.name, borrow=True)
class DoubleInputHiddenLayer(Layer):
    def __init__(self,  input1, input2, n_in1, n_in2, n_out, rng,activation=T.tanh,name="DoubleInputHiddenLayer",
                 params=None):
        self.n_in=[n_in1,n_in2]
        self.n_out=n_out
        self.activation=activation
        self.rng=rng
        super(DoubleInputHiddenLayer, self).__init__(name,params)   
        lin_output = T.dot(input1, self.W1)+T.dot(input2, self.W2)+ self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        self.L1 = abs(self.W1).sum()+abs(self.W2).sum()
        self.L2 = (self.W1 ** 2).sum() + (self.W2 ** 2).sum()
        self.params = [self.W1, self.W2, self.b]
        self.n_in=n_in1+n_in2
    
    def initParams(self):
        W1_values = np.asarray(self.rng.uniform(
    #       low=-np.sqrt(6. / (n_in + n_out)),
    #       high=np.sqrt(6. / (n_in + n_out)),
            low=-np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
            high=np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
    #       low=-np.sqrt(3. / np.sqrt(n_in)),
    #       high=np.sqrt(3. / np.sqrt(n_in)),
        size=(self.n_in[0], self.n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W1_values *= 4
        self.W1 = theano.shared(value=W1_values, name='W1_'+self.name, borrow=True)
        
        W2_values = np.asarray(self.rng.uniform(
    #       low=-np.sqrt(6. / (n_in + n_out)),
    #       high=np.sqrt(6. / (n_in + n_out)),
            low=-np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
            high=np.sqrt(1. / (self.n_in[0]+self.n_in[1])),
    #       low=-np.sqrt(3. / np.sqrt(n_in)),
    #       high=np.sqrt(3. / np.sqrt(n_in)),
        size=(self.n_in[1], self.n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W2_values *= 4
        self.W2 = theano.shared(value=W2_values, name='W2_'+self.name, borrow=True)
        
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_'+self.name, borrow=True)

class HiddenLayer(Layer):
    def __init__(self,  input, n_in, n_out, rng,activation=T.tanh,name="HiddenLayer",
                 params=None):
        self.n_in=n_in
        self.n_out=n_out
        self.activation=activation
        self.rng=rng
        super(HiddenLayer, self).__init__(name,params)   
        lin_output = T.dot(input, self.W)+ self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()
        self.params = [self.W, self.b]
    
    def initParams(self):
        W_values = np.asarray(self.rng.uniform(
    #       low=-np.sqrt(6. / (n_in + n_out)),
    #       high=np.sqrt(6. / (n_in + n_out)),
            low=-np.sqrt(1. / (self.n_in)),
            high=np.sqrt(1. / (self.n_in)),
    #       low=-np.sqrt(3. / np.sqrt(n_in)),
    #       high=np.sqrt(3. / np.sqrt(n_in)),
        size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
        if self.activation == theano.tensor.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value=W_values, name='W_'+self.name, borrow=True)
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_'+self.name, borrow=True)

class LogisticRegressionLayer(Layer):
    def __init__(self,input, n_in, n_out,rng,name="LogisticRegressionLayer",params=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.n_in=n_in
        self.n_out=n_out
        self.rng=rng
        super(LogisticRegressionLayer, self).__init__(name,params) 
        print input
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x =  T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.L1 = abs(self.W).sum()
        self.L2 = (self.W ** 2).sum()
        # parameters of the model
        self.params = [self.W, self.b]
        #self.n_in=1.0

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    def initParams(self):
        print self.n_in, self.n_out
        W_values = numpy.asarray(self.rng.uniform(
#            low=-numpy.sqrt(6. / (n_in + n_out)),
#            high=numpy.sqrt(6. / (n_in + n_out)),
            low=-numpy.sqrt(1. / (self.n_in)),
            high=numpy.sqrt(1. / (self.n_in)),
#            low=-numpy.sqrt(3. / numpy.sqrt(n_in)),
#            high=numpy.sqrt(3. / numpy.sqrt(n_in)),
            size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_'+self.name, borrow=True)
#            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
#                                                     dtype=theano.config.floatX),
#                                    name='W', borrow=True)
            # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((self.n_out,),
                                                 dtype=theano.config.floatX),
                                   name='b_'+self.name, borrow=True)
