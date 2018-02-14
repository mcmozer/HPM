'''
Modification of the theano demo code ('tweet sentiment analyzer'). Three
major changes were made:
(1) this code performs autoprediction in which each element of an input sequence
can be associated with a target output value. Consequently, the code can be
used to predict the next element in a sequence.
(2) Each input and output is tagged with an explicit time delay.
(3) The recurrent hidden layer can be either LSTM or HPM.
'''

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

import read_time_indexed_data

numpy.set_printoptions(threshold=numpy.nan)

#from theano.compile.debugmode import DebugMode

# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3

############################################################################
#theano.config.exception_verbosity = 'high' ################################
#theano.config.compute_test_value = 'raise' ################################
#theano.config.optimizer = 'None'           ################################
#theano.config.mode = 'FAST_COMPILE'        ################################
############################################################################

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed()  
# numpy.random.seed(SEED) to replicate

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not memory model) parameter. For the embeding and the classifier.
    """

    params = OrderedDict()
    # classifier
    if (options['arch_output_fn'] != '1-1'):
       params['U'] = 0.01 * numpy.random.randn(options['n_hid'],
       				  options['n_out']).astype(config.floatX)
       params['b'] = numpy.zeros((options['n_out'],)).astype(config.floatX)


    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name] # retrieves init and activation function
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

###############################################################################
# param_init_lstm
###############################################################################

def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid']),
                           ortho_weight(options['n_hid'])], axis=1)
    # augment LSTM input feature weights to include delta_t values
    if (options['arch_lstm_include_delta_t']):
       U = numpy.concatenate([U, 
                      .01 * numpy.random.randn(2,4*options['n_hid'])],axis=0)

    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['n_hid'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    # initialize embedding
    if (options['arch_remap_input']):
        randn = numpy.random.randn(options['n_in'],options['n_hid']);
        params['Wemb'] = (.01 * randn).astype(config.floatX)

    return params

###############################################################################
# lstm_layer
###############################################################################

def lstm_layer(tparams, state_below, xt, yt, options, prefix='lstm', mask=None):
    # xt and yt are used as additional inputs

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, delta_t_input, delta_t_output, h_, c_):
        # h_ has dim n_training_examples  X  n_lstm
        # delta_t_input has dim n_training_examples

        # include input and output delta_t values as LSTM input features
        if (options['arch_lstm_include_delta_t']):
            h_aug = tensor.concatenate([h_, delta_t_input[:,None], 
                                     delta_t_output[:,None]], axis=1)
        else:
            h_aug = h_

        preact = tensor.dot(h_aug, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['n_hid']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['n_hid']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['n_hid']))
        c = tensor.tanh(_slice(preact, 3, options['n_hid']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    n_hid = options['n_hid']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below, xt, yt],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           n_hid),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           n_hid)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

###############################################################################
# param_init_hpm
###############################################################################

def param_init_hpm(options, params, prefix='hpm'):
    """
    Init the HPM parameters:

    :see: init_params
    """
    n_hid = options['n_hid']

    if (options['arch_hpm_recurrent']):
       U = .01 * ortho_weight(n_hid) # DEBUG: what should scale of weights be?
       params[_p(prefix, 'U')] = U
       b = .05 * numpy.random.randn(n_hid,) - 1.5
       params[_p(prefix, 'b')] = b.astype(config.floatX)

    baserate = 1./10000.*numpy.ones((n_hid))  
    mu = -numpy.log(1-baserate) # mu = -log(1-baserate)
    params[_p(prefix, 'mu')] = numpy.log(
                      numpy.exp(mu.astype(config.floatX))-1.0) # anti soft-plus

    alpha = numpy.minimum(1.0, 
                 numpy.random.exponential(scale=0.2,size=n_hid)) # scale = mean
    params[_p(prefix, 'alpha')] = numpy.log(
                 numpy.exp(alpha.astype(config.floatX)) - 1.0) # anti soft-plus

    priorexp = 10.0**(-5.0*numpy.random.rand(n_hid))
    params[_p(prefix, 'priorexp')] = priorexp.astype(config.floatX)

    event_gamma_exp = numpy.log(numpy.exp(1.0)-1.0) * numpy.ones((n_hid)) 
                                                                # anti soft-plus
    params[_p(prefix, 'event_gamma_exp')] = event_gamma_exp.astype(config.floatX)

    # initialize embedding for sparsity
    if (options['arch_remap_input']):
        randn = numpy.random.rand(options['n_in'],options['n_hid'])
        params['Wemb'] = numpy.minimum(0.0, numpy.maximum(-4.0,
                        -numpy.log(1.0/randn.astype(config.floatX) - 1.0) ))
    return params

###############################################################################
# hpm_layer
###############################################################################

def hpm_layer(tparams, state_below, xt, yt, options, prefix='hpm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_examples = state_below.shape[1]
    else:
        n_examples = 1

    assert mask is not None

    n_hid = options['n_hid']
    timescales = options['timescales']
    gamma = (1.0 / numpy_floatX(timescales)).reshape((1, -1, 1))
    n_timescales = len(timescales)

    mu = (tensor.nnet.softplus(
			     tparams[_p(prefix, 'mu')])).dimshuffle(('x','x',0))

    # alpha is scaled by gamma to force longer time scales to have
    # smaller alpha values
    event_gamma_exp = (tensor.nnet.softplus(
                             tparams[_p(prefix, 'event_gamma_exp')])).dimshuffle(('x','x',0))
    alpha = (gamma ** event_gamma_exp) * (tensor.nnet.softplus(
                           tparams[_p(prefix,'alpha')])).dimshuffle(('x','x',0))

    def _timescale_posterior(likelihood, prior):
        # likelihood, prior, posterior have dimensions: 
        #               n_training_examples  X  n_timescales  X  n_hid
        # Make sure we don't crap out with 0 likelihoods
        posterior = prior * tensor.switch(
                    tensor.gt(tensor.max(likelihood,axis=1,keepdims=True), 0.0),
                    likelihood, likelihood + 1e-10)
        posterior = posterior / tensor.sum(posterior,axis=1,keepdims=True)
        return posterior

    def _marginalize_timescale(quantity, timescale_prob):
	q = quantity.dimshuffle([1,0,2]).flatten(ndim=2).dimshuffle([1,0])
	t = timescale_prob.dimshuffle([1,0,2]).flatten(ndim=2).dimshuffle([1,0])
        return tensor.batched_dot(q, t).reshape([n_examples, n_hid])

    def _event_prob(intensity, delta_t):
        gamma_exp = numpy.exp(-gamma * delta_t)   # 1 x timescales x 1
        gamma_factor = (1.0 - gamma_exp) / gamma  # 1 x timescales x 1
        prob = tensor.exp(-(intensity * gamma_factor + mu * delta_t)) 
        new_intensity = gamma_exp * intensity
        return new_intensity, prob

    def _step(m, state_below, delta_t_input, delta_t_output, h_, c_, yhat_):

        h, z  = _event_prob(h_, delta_t_input[:,None,None])

        # credit assignment across timescales
        c = _timescale_posterior(z, c_)

	if (options['arch_hpm_recurrent']):
            marginal_intensity_input = _marginalize_timescale(h + mu, c)
            # for recurrent connections, use updated intensity expectation
	    event = tensor.nnet.sigmoid(state_below 
		 + tensor.dot(marginal_intensity_input, tparams[_p(prefix,'U')])
		 + tparams[_p(prefix,'b')])
        elif (options['arch_remap_input']):
	    event = tensor.nnet.sigmoid(state_below + tparams[_p(prefix,'b')])
        else:
	    event = state_below

	# dimensions: n_training_examples X n_timescales X n_hid
	event = event.dimshuffle((0,'x',1))

        # credit assignment across timescales
        c = event * _timescale_posterior(h + mu,c) + (1.0-event) * c

        # update intensity 
	h += alpha * event

	# clear out updates after end of sequence
        c = m[:, None, None] * c + (1. - m)[:, None, None] * c_
        h = m[:, None, None] * h + (1. - m)[:, None, None] * h_

	# predict next event conditioned on timescale
        hhat, zhat = _event_prob(h, delta_t_output[:,None,None])
        chat = _timescale_posterior(zhat,c)

        # expectation of intensity 
        marginal_intensity_output = _marginalize_timescale(hhat + mu, chat)

	return h, c, marginal_intensity_output

    h = tensor.tensor3('h', dtype=config.floatX)
    # dimensions: n_training_examples  X  n_timescales  X  n_hid
    c = tensor.tensor3('c', dtype=config.floatX)
    # dimensions: n_training_examples  X  n_timescales  X  n_hid

    c0 = gamma ** tparams[_p(prefix,'priorexp')].dimshuffle(('x','x',0))
    c0 = c0 / tensor.sum(c0, axis=1)

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below, xt, yt],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), #h
                                                           n_examples,
							   n_timescales,
                                                           n_hid),
                                              tensor.alloc(c0, #c
                                                           n_examples,
							   n_timescales,
                                                           n_hid),
                                              tensor.alloc(numpy_floatX(0.),#yhat
                                                           n_examples,
                                                           n_hid)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[2], rval[0], rval[1] # return yhat, h, c


def sgd(lr, tparams, grads, x, xt, y, yt, mask, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, xt, y, yt, mask], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, xt, y, yt, mask, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, xt, y, yt, mask], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, xt, y, yt, mask, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, xt, y, yt, mask], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    # MIKE: why is this not a shared variable as in
    # trng = theano.tensor.shared_randomstreams.RandomStreams(1234)
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    xt = tensor.matrix('xt', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')
    yt = tensor.matrix('xt', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_examples = x.shape[1]

    if (options['arch_remap_input']):
       emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
						   n_examples,
						   options['n_hid']])
    else:
       Wemb = theano.shared( numpy.concatenate(
                      (numpy.zeros((1,options['n_hid']),dtype=config.floatX),
                       numpy.identity(options['n_hid'],dtype=config.floatX)),
                       axis=0), name='Wemb')
       emb = Wemb[x.flatten()].reshape([n_timesteps,
                                        n_examples,
                                        options['n_hid']])

    # this is the call to either lstm_layer or hpm_layer
    if (options['encoder'] == 'lstm'):
        proj = get_layer(options['encoder'])[1](
                                            tparams, emb, xt, yt, options,
                                            prefix=options['encoder'],
                                            mask=mask)
        h = c = None
    else:
        proj, h, c = get_layer(options['encoder'])[1](
                                            tparams, emb, xt, yt, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    # proj has dim n_timesteps X n_examples X n_hid
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
    def _step(proj_step):
       if (options['arch_output_fn'] == 'softmax'):
          pred_prob_step = tensor.nnet.softmax(
                            tensor.dot(proj_step, tparams['U']) + tparams['b'])
       elif (options['arch_output_fn'] == 'logistic'):
          pred_prob_step = tensor.nnet.sigmoid(
                            tensor.dot(proj_step, tparams['U']) + tparams['b'])
       else: # '1-1'
          pred_prob_step = proj_step / tensor.sum(proj_step,axis=1,keepdims=True)
	  pred_prob_step = tensor.concatenate([tensor.alloc(0,n_examples,1),
	                                       pred_prob_step], axis=1)
       return pred_prob_step
       # pred_prob_step should have dim n_examples X n_outputs
       # pred_prob has dim n_timesteps x n_examples x n_outputs
       # pred_step has have dim n_examples 

    pred_prob, updates = theano.scan(_step,
                                sequences=proj,
				outputs_info=None,
				non_sequences=None,
				n_steps=n_timesteps)

    def _cost_step_norm(pred_prob_step, y_step):
        # tgt_prob_step should have dim n_examples 
        tgt_prob_step = tensor.switch(tensor.eq(y_step, 0), 1.0, 
                             pred_prob_step[tensor.arange(n_examples),y_step]/
			     (1.0-pred_prob_step[tensor.arange(n_examples),0]))

        # need to add 1 to pass by index 0 which we removed in computing max
        pred_ix_step = tensor.argmax(pred_prob_step[:,1:],axis=1) + 1
        if (options['type_token_sim']): 
            corr_step = tensor.switch(tensor.eq(y_step, 0), 0,
                      tensor.switch(tensor.eq((y_step-1)//5, 
                                     (pred_ix_step-1)//5), 1, -1))
        else:
            corr_step = tensor.switch(tensor.eq(y_step, 0), 0,
                      tensor.switch(tensor.eq(y_step,pred_ix_step), 1, -1))
        return tgt_prob_step, corr_step

    # cost function for predicting target value of a specific event
    # tgt_prob_step should have dim n_examples
    def _cost_step_tgt(pred_prob_step, y_step):
        tgt_prob_step = tensor.switch(tensor.eq(y_step, 0), 1.0, 
                     tensor.switch(tensor.gt(y_step, 0),
                        pred_prob_step[tensor.arange(n_examples),y_step],
                        1.0-pred_prob_step[tensor.arange(n_examples),-y_step]))
        corr_step = tensor.switch(tensor.eq(y_step, 0), 0,
                     tensor.switch(tensor.gt(tgt_prob_step, 0.5), 1, -1))
        return tgt_prob_step, corr_step

    if (options['signed_out']):
        cost_fn = _cost_step_tgt
    else:
        cost_fn = _cost_step_norm

    (tgt_prob, corr), updates = theano.scan(cost_fn,
                                            sequences=[pred_prob, y],
                                            outputs_info=None,
                                            non_sequences=None,
                                            n_steps=n_timesteps)

    off = 1e-8
    if tgt_prob.dtype == 'float16':
        off = 1e-6
    # tgt_prob: probability correct (dimensions n_timesteps X n_examples)
    cost = -tensor.sum(tensor.log(tgt_prob.clip(off, 1.0))) 
    # Note: not dividing by count because it will reweight minibatch by size
    #         / tensor.sum(tensor.gt(y,0))

    return use_noise, x, xt, y, yt, mask, pred_prob, corr, cost, proj, h, c


# This code is currently not being used
#def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
#    """ If you want to use a trained model, this is useful to compute
#    the probabilities of new examples.
#    """
#    n_examples = len(data[X])
#    probs = numpy.zeros((n_examples, 2)).astype(config.floatX)
#
#    n_done = 0
#
#    for _, trial_ix in iterator:
#        x, xt, y, yt, mask = prepare_data([data[X][t] for t in trial_ix],
#                                          [data[XT][t] for t in trial_ix],
#                                          [data[Y][t] for t in trial_ix],
#                                          [data[YT][t] for t in trial_ix],
#                                          maxlen=None)
#        # x, xt, y, yt, mask:  dimensions max_steps X n_examples
#        pred_probs = f_pred_prob(x, xt, yt, mask)
#        probs[trial_ix, :] = pred_probs
#
#        n_done += len(trial_ix)
#        if verbose:
#            print('%d/%d examples classified' % (n_done, n_examples))
#
#    return probs
#


def pred_error(f_corr, f_cost, prepare_data, data, iterator, options, verbose=False):
    """
    Just compute the error
    f_corr: Theano function to compute the accuracy for a minibatch
    f_cost: Theano function to compute the cost of a minibatch
    prepare_data: usual prepare_data for that dataset.
    """
    correct_ct = all_ct = 0
    cost = 0.0
    for _, trial_ix in iterator:
        x, xt, y, yt, mask = prepare_data([data[X][t] for t in trial_ix],
	                                  [data[XT][t] for t in trial_ix],
	                                  [data[Y][t] for t in trial_ix],
	                                  [data[YT][t] for t in trial_ix],
                                          maxlen=None)
	cost += f_cost(x, xt, y, yt, mask) 
        corr = f_corr(x, xt, y, yt, mask)
        correct_ct += (corr > 0).sum()
        all_ct += (corr != 0).sum()
    err = 1.0 - numpy_floatX(correct_ct) / numpy_floatX(all_ct)
    cost = cost / numpy_floatX(all_ct)

    return err, cost


def print_weights(tparams,options):
    print_params = unzip(tparams)
    if (options['encoder'] == 'hpm'):
        print('hpm_mu:\n',numpy.log(1.0+numpy.exp(print_params['hpm_mu'])))
        print('hpm_alpha:\n',numpy.log(
                                      1.0+numpy.exp(print_params['hpm_alpha'])))
        print('hpm_prior_exp:\n', print_params['hpm_priorexp'])
    if (options['arch_output_fn'] != '1-1'):
        print('U:\n',print_params['U'])
        print('b:\n',print_params['b'])
    if (options['arch_remap_input']):
        print('Wemb:\n',print_params['Wemb'])

def recode_value_as_index(data):
    new_data = [[abs(data[n][j])*2-(data[n][j]<0) 
                   for j in range(len(data[n]))] for n in range(len(data))]
    return new_data

def train_model(
    arch_remap_input=True, # should input layer be remapped to hidden or 1-1?
    arch_output_fn='softmax', # 'softmax', 'logistic', or '1-1'
    #arch_hpm_event_lik='m', # 'e'=expectation, 'b'=bernoulli, 'm'=marginal
    arch_hpm_recurrent=False, # include recurrent connections between memory units
    arch_lstm_include_delta_t = True, #include delta_t values in lstm input 
    n_hid=40,  # number of HPM hidden units; ignored if arch_remap_input True
    patience=10,  # Number of epoch to wait before early stop if no progress
                  # -1 = no early stopping
    save_freq=-1,  # Save the parameters after every save_freq updates
                   # -1 = no save
    valid_freq=5,  # Compute the validation error after this number of update.
    max_epochs=1500,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentum and decaying learning rate).
    encoder='hpm',  # lstm or hpm
    saveto='hpm_model.npz',  # The best model will be saved there
    maxlen=500,  # Sequence longer then this get ignored
    batch_size=16, # The batch size during training.
    valid_batch_size=16,  # The batch size used for validation/test set.
    data_file='../data/type_token/hp_4streams_5tokens_1',
    timescales=[1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024.],
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    valid_portion=.10, # proportion of training examples for validation
    show_weights = False, # show weights every valid_freq epochs
    type_token_sim = False # sim that lumps together groups of 5 outputs 
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    global layers
    if model_options['encoder'] == 'lstm':
        layers = {'lstm': (param_init_lstm, lstm_layer)}
    elif model_options['encoder'] == 'hpm':
        layers = {'hpm': (param_init_hpm, hpm_layer)}
    else:
        sys.exit("Invalid encoder");

    load_data = read_time_indexed_data.load_data;
    prepare_data = read_time_indexed_data.prepare_data;

    if (model_options['type_token_sim']):
       print('WARNING: type_token_sim flag is set')
       # type_token_sim is Mike's hack to lump together tokens of the
       # same type when computing model accuracy

    print('Loading data')
    train, valid, test = load_data(data_file, valid_portion=valid_portion, 
                                   maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[X]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[X][n] for n in idx], 
	        [test[XT][n] for n in idx],
	        [test[Y][n] for n in idx],
	        [test[YT][n] for n in idx])
    # if the input or output has negative indices, then the sign of each
    # index should be interpreted as a target value. On the input, the number
    # of units will be doubled and will be indexed in the order: 
    # 0, -1, +1, -2, +2, etc. On the output, the sign will indicate a target
    # value for the index (-1 -> 0, +1 -> 1)

    model_options['signed_in'] = numpy.min(list(chain(*train[X]))) < 0
    if (model_options['signed_in']):
       train = (recode_value_as_index(train[X]), train[XT], train[Y], train[YT])
       valid = (recode_value_as_index(valid[X]), valid[XT], valid[Y], valid[YT])
       test = (recode_value_as_index(test[X]), test[XT], test[Y], test[YT])

    model_options['n_in'] = numpy.max(list(chain(*train[X]))) + 1 
    model_options['signed_out'] = numpy.min(list(chain(*train[Y]))) < 0
    model_options['n_out'] = numpy.max(numpy.absolute(list(chain(*train[Y])))) + 1 
    if (model_options['signed_out']):
        model_options['arch_output_fn'] = 'logistic'
    if (model_options['arch_output_fn'] != '1-1' and
                            model_options['arch_output_fn'] != 'logistic' and
                            model_options['arch_output_fn'] != 'softmax'):
        print('Error: invalid arch_output_fn')
        quit()
       

    # if 1-1 mapping between inputs and hidden, then n_hid is forced to 
    # input max
    if (not model_options['arch_remap_input'] and
                                               model_options['arch_hpm_recurrent']):
       print('Error: recurrent arch only if input remapped');
       quit()
    if (not model_options['arch_remap_input'] 
                and model_options['arch_output_fn']=='1-1'
                and model_options['n_out'] != model_options['n_in']):
       print('Error: # input and output alternatives must match');
       quit()
    if (not model_options['arch_remap_input']):
       model_options['n_hid'] = model_options['n_in']-1
    if (model_options['arch_output_fn'] == '1-1'):
       model_options['n_hid'] = model_options['n_out']-1

    print('Building model with',model_options['n_in']-1,'inputs,',
          model_options['n_hid'],'hidden, and',model_options['n_out']-1,
          ['','signed'][model_options['signed_out']],'outputs')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('hpm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, xt, y, yt, mask, pred_prob, corr, cost, 
                              proj, h, c) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, xt, y, yt,mask], cost, name='f_cost')

    # selection of index with strongest prediction
    f_corr = theano.function([x, xt, y, yt, mask], corr, name='f_corr')

    # probability distribution over alternatives
    f_pred_prob = theano.function([x, xt, yt, mask], 
                                              pred_prob, name='f_pred_prob')
    # cost function for predicting one-of-N poassible events

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, xt, y, yt, mask], grads, name='f_grad')

    # DEBUG output functions
    #f_h = theano.function([x, xt, yt, mask], h, name='f_h')
    #f_c = theano.function([x, xt, yt, mask], c, name='f_c')
    #f_proj = theano.function([x, xt, yt, mask], proj, name='f_proj')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, xt, y, yt, mask, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[X]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[X]), valid_batch_size)

    print("%d train examples" % len(train[X]))
    print("%d valid examples" % len(valid[X]))
    print("%d test examples" % len(test[X]))

    history = []
    best_p = None

    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(1,max_epochs+1):
            n_examples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[X]), batch_size, shuffle=True)

            for _, train_index in kf:
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [train[X][t] for t in train_index]
                xt = [train[XT][t] for t in train_index]
                y = [train[Y][t] for t in train_index]
                yt = [train[YT][t] for t in train_index]


                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n examples)
                x, xt, y, yt, mask = prepare_data(x, xt, y, yt)
                n_examples += x.shape[1]

                # DEBUG
                #h = f_h(x, xt, yt, mask)
                #c = f_c(x, xt, yt, mask)
                #proj = f_proj(x, xt, yt, mask)
                #print('h (#steps X # examples X #scales X #hid):\n', h)
                #print('c (#steps X # examples X #scales X #hid):\n', c)
                #print('proj (#steps X # examples X #hid):\n', proj)

                cost = f_grad_shared(x, xt, y, yt, mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
		    print_weights(tparams, model_options)
                    return 1., 1., 1.

            # end minibatch loop

	    if numpy.mod(eidx, valid_freq) == 0:
		use_noise.set_value(0.)
		train_err, train_cost = pred_error(f_corr,f_cost, 
                                            prepare_data,train,kf,model_options)
		valid_err, valid_cost = pred_error(f_corr, f_cost, 
                                      prepare_data,valid,kf_valid,model_options)
		test_err, test_cost = pred_error(f_corr,f_cost, 
                                        prepare_data,test,kf_test,model_options)

		print('%4d: Acc (Cost): Tr %7.5f (%7.5f) Va %7.5f (%7.5f) Te %7.5f (%7.5f)' 
		      % (eidx, 1-train_err, train_cost, 1-valid_err, 
                         valid_cost, 1-test_err, test_cost) )

		if (best_p is None or
			     valid_cost <= numpy.array(history)[:,0].min()):
		    best_p = unzip(tparams)
		    bad_counter = 0

		if (patience > 0 and len(history) >= patience and
                        valid_cost >= numpy.array(history)[-patience:,0].min()):
                        # was valid_cost >= numpy.array(history)[:-patience,0].min()):
		    bad_counter += 1
		    if bad_counter > patience:
		        print('Early Stop!')
		        estop = True
		        break

		history.append([valid_cost, valid_err, test_cost, test_err])
                if (model_options['show_weights']):
                    print_weights(tparams, model_options)

	    if saveto and save_freq > 0 and numpy.mod(eidx, save_freq) == 0:
		print('Saving...')

		if best_p is not None:
		    params = best_p
		else:
		    params = unzip(tparams)
		numpy.savez(saveto, history=history, **params)
		pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
        # end loop eidx over (max_epochs):
	#if estop:
	#    break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[X]), batch_size)
    train_err, train_cost = pred_error(f_corr,f_cost,prepare_data,
                                            train,kf_train_sorted,model_options)
    valid_err, valid_cost = pred_error(f_corr,f_cost,prepare_data, 
                                                   valid,kf_valid,model_options)
    test_err, test_cost  = pred_error(f_corr,f_cost,prepare_data, 
                                                     test,kf_test,model_options)

    print('%4d: Acc (Cost): Tr %7.5f (%7.5f) Va %7.5f (%7.5f) Te %7.5f (%7.5f)' 
	  % (eidx, 1-train_err, train_cost, 1-valid_err, valid_cost, 
	     1-test_err, test_cost) )
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history=history, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)

    if (model_options['show_weights']):
        print_weights(tparams, model_options)
    return train_err, valid_err, test_err
