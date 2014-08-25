#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""util.py: Useful functions"""

from optparse import OptionParser
import logging
import sys
from io import open
from os import path
from collections import defaultdict
import re
import signal
import numpy
import theano.tensor as T

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def zero_value(shape, type):
  return numpy.zeros(shape, dtype=type)

def random_value(shape, type, random_generator=None):
  """
  Return a randomly initialized matrix.

  :type rng: numpy.random.RandomState
  :param rng: a random number generator used to initialize weights
  """
  # `W` is initialized with `W_values` which is uniformely sampled
  # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
  # for tanh activation function
  # the output of uniform if converted using asarray to dtype
  # theano.config.floatX so that the code is runable on GPU
  # Note : optimal initialization of weights is dependent on the
  #        activation function used (among other things).
  #        For example, results presented in [Xavier10] suggest that you
  #        should use 4 times larger initial weights for sigmoid
  #        compared to tanh
  #        We have no info for other function, so we use the same as
  #        tanh.
  if not random_generator:
    random_generator = numpy.random.RandomState(1234)
  total_dimensions = numpy.sum(shape)
  print shape
  low = -numpy.sqrt(6./total_dimensions)
  high = numpy.sqrt(6./total_dimensions)
  random_values = random_generator.uniform(low=low, high=high, size=shape)
  W_values = numpy.asarray(random_values, dtype=type)
  return W_values

def hardtanh(x):
    return x**(T.abs_(x)<1)*(-1)**(x<=-1)