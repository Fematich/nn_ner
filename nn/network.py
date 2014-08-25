# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import itertools
import theano.tensor as T
import numpy as np

from layers import EmbeddingLayer, StaticEmbeddingLayer, DoubleInputHiddenLayer, LogisticRegressionLayer
from layers import DoubleEmbeddingHiddenLayer
from interfaces import Network
from util import hardtanh

class oldennaNER(Network):
    def __init__(self, input,embeddings,mini_batch_size=32,nhu=300,width=5,activation=hardtanh,seed=1234,n_out=9,name='SennaNER',params=None):
        self.name = name
        self.layers = []
        self.input = input
        self.output = None

        embedding_dim=embeddings.shape[1]
        features = np.eye(4)

        rng=np.random.RandomState(seed)
        self.EmbeddingLayer = EmbeddingLayer(input=input[:,:,0],w_values=embeddings,embedding_dim=embedding_dim,mini_batch_size=mini_batch_size,width=width,params=params)
        self.StaticEmbeddingLayer = StaticEmbeddingLayer(input=input[:,:,1],w_values=features,embedding_dim=features.shape[0],mini_batch_size=mini_batch_size,width=width)
        self.HiddenLayer = DoubleInputHiddenLayer(input1=self.EmbeddingLayer.output, input2=self.StaticEmbeddingLayer.output, n_in1=embedding_dim*width, n_in2=features.shape[0]*width, n_out=nhu, rng=rng, activation=activation,params=params)
        self.LogisticRegressionLayer = LogisticRegressionLayer(input=self.HiddenLayer.output,n_in=nhu,n_out=n_out, rng=rng, params=params)
        self.layers=[self.EmbeddingLayer,self.StaticEmbeddingLayer,self.HiddenLayer,self.LogisticRegressionLayer]

        self.L1 = T.sum([layer.L1 for layer in self.layers if "L1" in layer.__dict__])
        self.L2 = T.sum([layer.L2 for layer in self.layers if "L2" in layer.__dict__])

        self.params = list(itertools.chain(*[layer.params for layer in self.layers]))
       
        self.negative_log_likelihood = self.LogisticRegressionLayer.negative_log_likelihood
        self.errors = self.LogisticRegressionLayer.errors
        self.predictions = self.LogisticRegressionLayer.y_pred
        self.n_ins = list(itertools.chain(*[[layer.n_in]*len(layer.params) for layer in self.layers]))
        print self.n_ins
        print self.params

class SennaNER(Network):
    def __init__(self, input,embeddings,mini_batch_size=32,nhu=300,width=5,activation=hardtanh,seed=1234,n_out=9,name='SennaNER',params=None):
        self.name = name
        self.layers = []
        self.input = input
        self.output = None

        embedding_dim=embeddings.shape[1]
        features = np.eye(4)

        rng=np.random.RandomState(seed)
        self.HiddenLayer = DoubleEmbeddingHiddenLayer(input=input,embeddings_values=embeddings,features_values=features,n_in1=embedding_dim*width,n_in2=features.shape[0]*width, batch_size=mini_batch_size, n_out=nhu, rng=rng, activation=activation,params=params)
#        self.EmbeddingLayer = EmbeddingLayer(input=input[:,:,0],w_values=embeddings,embedding_dim=embedding_dim,mini_batch_size=mini_batch_size,width=width,params=params)
#        self.StaticEmbeddingLayer = StaticEmbeddingLayer(input=input[:,:,1],w_values=features,embedding_dim=features.shape[0],mini_batch_size=mini_batch_size,width=width)
#        self.HiddenLayer = DoubleInputHiddenLayer(input1=self.EmbeddingLayer.output, input2=self.StaticEmbeddingLayer.output, n_in1=embedding_dim*width, n_in2=features.shape[0]*width, n_out=nhu, rng=rng, activation=activation,params=params)
        self.LogisticRegressionLayer = LogisticRegressionLayer(input=self.HiddenLayer.output,n_in=nhu,n_out=n_out, rng=rng, params=params)
        self.layers=[self.HiddenLayer,self.LogisticRegressionLayer]

        self.L1 = T.sum([layer.L1 for layer in self.layers if "L1" in layer.__dict__])
        self.L2 = T.sum([layer.L2 for layer in self.layers if "L2" in layer.__dict__])

        self.params = list(itertools.chain(*[layer.params for layer in self.layers]))
       
        self.negative_log_likelihood = self.LogisticRegressionLayer.negative_log_likelihood
        self.errors = self.LogisticRegressionLayer.errors
        self.predictions = self.LogisticRegressionLayer.y_pred
        self.n_ins = list(itertools.chain(*[[layer.n_in]*len(layer.params) for layer in self.layers]))
        print self.n_ins
        print self.params