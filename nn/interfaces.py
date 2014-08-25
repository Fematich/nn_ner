# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import theano

class Layer(object):
    def __init__(self,name, params=None):
        self.name=name
        self.input = input
        self.params = []
        if params!=None:
            self.setParams(params=params.__dict__.get(name))
        else:
            self.initParams()

    def __getstate__(self):
        params = {}
        for param in self.params:
            params[param.name] = param.get_value()
        return params

    def setParams(self,params):
        for pname,param in params.__dict__.iteritems():
            self.__dict__[pname[:-(len(self.name)+1)]] = theano.shared(param, name=pname[:-(len(self.name)+1)]+'_'+self.name, borrow=True)

    def initParams():
        raise NotImplementedError

class Network():
    def __getstate__(self):
        return dict([(layer.name,layer) for layer in self.layers])
