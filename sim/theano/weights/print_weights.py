from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from itertools import chain

numpy.set_printoptions(threshold=numpy.nan)

#*************************************************************
model = '0102' # or '0416A'
file_name = 'reddit/hpm_0102_testAAA.npz'
#*************************************************************

with numpy.load(file_name) as data:
#with numpy.load('./lstm_model.npz') as data:
#with numpy.load('./hpm_model_with_all_cheats.npz') as data:
    for k in  data:
        if (model == '0416A'):
            if (k == 'hpm_alpha' or k == 'hpm_mu'):
                print ('***** softplus ',k, '\n', numpy.log(1.0+numpy.exp(data[k])))
            else:
                print ('***** ',k, '\n', data[k])
        if (model == '1110'):
            if (k == 'hpm_mu' or k == 'hpm_gamma_exp_for_mu' or k == 'hpm_eta' or k == 'hpm_output_calibration'):
                print ('***** softplus ',k, '\n', numpy.log(1.0+numpy.exp(data[k])))
            elif (k == 'hpm_alpha0' or k == 'hpm_gamma_exp_for_alpha'):
                print ('***** logistic ',k, '\n', 1.0/(1.0 + numpy.exp(data[k])))
            else:
                print ('***** ',k, '\n', data[k])
        if (model == '0102'):
            if (k == 'hpm_mu' or k == 'hpm_gamma_exp_for_mu' or k == 'hpm_output_calibration'):
                print ('***** softplus ',k, '\n', numpy.log(1.0+numpy.exp(data[k])))
            else:
                print ('***** ',k, '\n', data[k])

