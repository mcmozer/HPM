'''
Build a tweet sentiment analyzer
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

import music


# order of information from load_data
X = 0
XT = 1
Y = 2
YT = 3

###########################################################################
#theano.config.exception_verbosity = 'high' # DEBUG ************************
#theano.config.compute_test_value = 'raise' # DEBUG ************************
#theano.config.optimizer = 'None'           # DEBUG ************************
###########################################################################

datasets = {'music': (music.load_data, music.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
#numpy.random.seed(SEED) DEBUG
#numpy.random.seed()

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


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


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

    # embedding: initialize for sparsity
    # DEBUG

    #if (options['arch_remap_input']):
    #   randn = numpy.random.randn(options['xdim'],
    #                              options['n_hpu'])
    #   params['Wemb'] = (0.01 * randn).astype(config.floatX)

    if (options['arch_remap_input']):
       randn = numpy.random.rand(options['xdim'],
                                 options['n_hpu'])
       params['Wemb'] = numpy.minimum(0.0, numpy.maximum(-4.0,
                          -numpy.log(1.0/randn.astype(config.floatX) - 1.0) ))

    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    if (options['arch_remap_output']):
       params['U'] = 0.01 * numpy.random.randn(options['n_hpu'],
       				  options['ydim']).astype(config.floatX)
       params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

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
# param_init_hpm
###############################################################################

def param_init_hpm(options, params, prefix='hpm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    n_hpu = options['n_hpu']

    if (options['arch_recurrent']):
       U = .01 * ortho_weight(n_hpu) # DEBUG: what should scale of weights be?
       params[_p(prefix, 'U')] = U
       b = numpy.zeros((n_hpu,))
       params[_p(prefix, 'b')] = b.astype(config.floatX)

    # mu = -log(1-baserate)
    baserate = 1e-6
    logmu = numpy.log(-numpy.log(1-baserate)) * numpy.ones((n_hpu))
    params[_p(prefix, 'logmu')] = logmu.astype(config.floatX)
    logalpha = numpy.log(0.5) * numpy.ones((n_hpu))
    params[_p(prefix, 'logalpha')] = logalpha.astype(config.floatX)
    logpriorexp = numpy.log(.0001) * numpy.ones((n_hpu))
    params[_p(prefix, 'logpriorexp')] = logpriorexp.astype(config.floatX)

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

    n_hpu = options['n_hpu']
    timescales = options['timescales']
    gamma = (1.0 / numpy.asarray(timescales)).reshape((1, -1, 1))
    n_timescales = len(timescales)
    mu = (tensor.exp(tparams[_p(prefix, 'logmu')])).dimshuffle(('x','x',0)) 
    # NOTE: alpha is scaled by gamma to force longer time scales to have
    # smaller alpha values
    alpha = gamma * (tensor.exp(tparams[_p(prefix,'logalpha')])).dimshuffle(('x','x',0))

    def _timescale_posterior(likelihood, prior):
        posterior = likelihood * prior
        posterior = posterior / tensor.sum(posterior,axis=1).dimshuffle((0,'x',1));
        return posterior

    def _marginalize_timescale(intensity, timescale_prob):
	i = intensity.dimshuffle([1,0,2]).flatten(ndim=2).dimshuffle([1,0])
	t = timescale_prob.dimshuffle([1,0,2]).flatten(ndim=2).dimshuffle([1,0])
        return tensor.batched_dot(i, t).reshape([n_examples, n_hpu])

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

        marginal_intensity_input = _marginalize_timescale(h + mu, c)

	if (options['arch_recurrent']):
           # for recurrent connections, use updated intensity expectation
	   event = tensor.nnet.sigmoid(state_below 
		 + tensor.dot(marginal_intensity_input, tparams[_p(prefix,'U')])
		 + tparams[_p(prefix,'b')])
        elif (options['arch_remap_input']):
	   event = tensor.nnet.sigmoid(state_below)
        else:
	   event = state_below

	# dimensions: #training examples X #timescales X #hpu
	event = event.dimshuffle((0,'x',1))

	# credit assignment across timescales
        c = _timescale_posterior(tensor.pow(h + mu, event),c)

        # update intensity 
	h = h + alpha * event

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
    # dimensions: # training examples  X  # timescales  X  # hpu
    c = tensor.tensor3('c', dtype=config.floatX)
    # dimensions: # training examples  X  # timescales  X  # hpu

    c0 = gamma ** (tensor.exp(tparams[_p(prefix,'logpriorexp')]).dimshuffle(('x','x',0)))
    c0 = c0 / tensor.sum(c0, axis=1)

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below, xt, yt],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), #h
                                                           n_examples,
							   n_timescales,
                                                           n_hpu),
                                              tensor.alloc(c0, #c
                                                           n_examples,
							   n_timescales,
                                                           n_hpu),
                                              tensor.alloc(numpy.asarray(0.),#yhat
                                                           n_examples,
                                                           n_hpu)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[2] # return yhat

# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
if 0:
    layers = {'lstm': (param_init_lstm, lstm_layer)}
else:
    layers = {'hpm': (param_init_hpm, hpm_layer)}


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
						   options['n_hpu']])
    else:
       Wemb = theano.shared( numpy.concatenate(
                      (numpy.zeros((1,options['n_hpu']),dtype=config.floatX),
                       numpy.identity(options['n_hpu'],dtype=config.floatX)),
                       axis=0), name='Wemb')
       emb = Wemb[x.flatten()].reshape([n_timesteps,
                                        n_examples,
                                        options['n_hpu']])

    # this is the call to either lstm_layer or hpm_layer
    proj = get_layer(options['encoder'])[1](tparams, emb, xt, yt, options,
                                            prefix=options['encoder'],
                                            mask=mask)

    # proj has dim n_timesteps X n_examples X n_hpu
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
    def _step(proj_step):
       if (options['arch_remap_output']):
          pred_prob_step = tensor.nnet.softmax(
                            tensor.dot(proj_step, tparams['U']) + tparams['b'])
       else:
          pred_prob_step = proj_step / tensor.sum(proj_step,axis=1,keepdims=True)
	  pred_prob_step = tensor.concatenate([tensor.alloc(0,n_examples,1),
	                                       pred_prob_step], axis=1)

       # need to add 1 to pass by index 0 which we removed in computing max
       pred_step = tensor.argmax(pred_prob_step[:,1:],axis=1) + 1
       return pred_prob_step, pred_step
       # pred_prob_step should have dim n_examples X n_outputs
       # pred_prob has dim n_timesteps x n_examples x n_outputs
       # pred_step has have dim n_examples 

    (pred_prob, pred), updates = theano.scan(_step,
                                sequences=proj,
				outputs_info=None,
				non_sequences=None,
				n_steps=n_timesteps)

    # probability distribution over alternatives
    f_pred_prob = theano.function([x, xt, yt, mask], pred_prob, name='f_pred_prob')
    # selection of index with strongest prediction
    f_pred = theano.function([x, xt, yt, mask], pred, name='f_pred')

    def _cost_step(pred_prob_step, y_step):
       tgt_prob_step = tensor.switch(tensor.eq(y_step, 0), 1.0, 
                             pred_prob_step[tensor.arange(n_examples),y_step]/
			     (1.0-pred_prob_step[tensor.arange(n_examples),0]))
       return tgt_prob_step
       # tgt_prob_step should have dim n_examples 

    tgt_prob, updates = theano.scan(_cost_step,
                                sequences=[pred_prob, y],
				outputs_info=None,
				non_sequences=None,
				n_steps=n_timesteps)

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    # tgt_prob: probability that non-zero target is predicted
    # dimensions -- # timesteps * # examples
    cost = -tensor.sum(tensor.log(tgt_prob.clip(off, 1.0))) 
    # Note: not dividing by count because it will reweight minibatch by size
    #         / tensor.sum(tensor.gt(y,0))

    return use_noise, x, xt, y, yt, mask, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_examples = len(data[X])
    probs = numpy.zeros((n_examples, 2)).astype(config.floatX)

    n_done = 0

    for _, trial_ix in iterator:
        x, xt, y, yt, mask = prepare_data([data[X][t] for t in trial_ix],
                                          [data[XT][t] for t in trial_ix],
                                          [data[Y][t] for t in trial_ix],
                                          [data[YT][t] for t in trial_ix],
                                          maxlen=None)
        # x, xt, y, yt, mask:  dimensions max_steps X n_examples
        pred_probs = f_pred_prob(x, xt, yt, mask)
        probs[trial_ix, :] = pred_probs

        n_done += len(trial_ix)
        if verbose:
            print('%d/%d examples classified' % (n_done, n_examples))

    return probs


def pred_error(f_pred, f_cost, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano function to compute the prediction for a minibatch
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
        preds = f_pred(x, xt, yt, mask)
	cost += f_cost(x, xt, y, yt, mask) 
        correct_ct += (numpy.logical_and(y>0, y== preds)).sum()
	all_ct += (y > 0).sum()
    err = 1.0 - numpy_floatX(correct_ct) / numpy_floatX(all_ct)
    cost = cost / numpy_floatX(all_ct)

    return err, cost


def print_weights(tparams,options):
    if (options['show_weights']):
       print_params = unzip(tparams)
       print('hpm_mu:\n',numpy.exp(print_params['hpm_logmu']))
       print('hpm_alpha:\n',numpy.exp(print_params['hpm_logalpha']))
       print('hpm_prior_exp:\n',numpy.exp(print_params['hpm_logpriorexp']))
       if (options['arch_remap_output']):
	  print('U:\n',print_params['U'])
	  print('b:\n',print_params['b'])
       if (options['arch_remap_input']):
	  print('Wemb:\n',print_params['Wemb'])


def train_model(
    arch_remap_input=True, # should input layer be remapped to hidden or 1-1?
    arch_remap_output=True, # should hidden layer be remapped to a softmax layer or 1-1 to output?
    arch_recurrent=False, # include recurrent connections between memory units
    n_hpu=40,  # number of HPM hidden units; ignored if arch_remap_input True
    patience=-1,  # Number of epoch to wait before early stop if no progress
                  # -1 = no early stopping
    save_freq=-1,  # Save the parameters after every save_freq updates
                   # -1 = no save
    valid_freq=5,  # Compute the validation error after this number of update.
    max_epochs=5000,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentum and decaying learning rate).
    encoder='hpm',  # lstm or hpm
    saveto='hpm_model.npz',  # The best model will be saved there
    maxlen=500,  # Sequence longer then this get ignored
    batch_size=16, # The batch size during training.
    valid_batch_size=16,  # The batch size used for validation/test set.
    dataset='music',
    timescales=[1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024.],
    noise_std=0.,
    use_dropout=False,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    valid_portion=.10, # proportion of training examples for validation
    show_weights = False, # show weights every valid_freq epochs
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data(valid_portion=valid_portion, maxlen=maxlen)
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
    # for both xdim and ydim we count index 0 as a possible no-op input or
    # output
    model_options['ydim'] = numpy.max(list(chain(*train[Y]))) + 1 
    model_options['xdim'] = numpy.max(list(chain(*train[X]))) + 1 
    # if 1-1 mapping between inputs and hidden, then n_hpu is forced to 
    # input max
    if (not model_options['arch_remap_input'] and
                                               model_options['arch_recurrent']):
       print('Error: recurrent arch only if input remapped');
       quit()
    if (not model_options['arch_remap_input'] 
                and not model_options['arch_remap_output']
                and model_options['ydim'] != model_options['xdim']):
       print('Error: # input and output alternatives must match');
       quit()
    if (not model_options['arch_remap_input']):
       model_options['n_hpu'] = model_options['xdim']-1
    if (not model_options['arch_remap_output']):
       model_options['n_hpu'] = model_options['ydim']-1

    print('Building model with',model_options['xdim']-1,'inputs,',
          model_options['n_hpu'],'hidden, and',model_options['ydim']-1,
	  'outputs')
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
    (use_noise, x, xt, y, yt, mask, 
     f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, xt, y, yt,mask], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, xt, y, yt, mask], grads, name='f_grad')

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

                cost = f_grad_shared(x, xt, y, yt, mask)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

            # end minibatch loop

	    if numpy.mod(eidx, valid_freq) == 0:
		use_noise.set_value(0.)
		train_err, train_cost = pred_error(f_pred, f_cost, 
                                                   prepare_data, train, kf)
		valid_err, valid_cost = pred_error(f_pred, f_cost, 
                                                  prepare_data, valid, kf_valid)
		test_err, test_cost = pred_error(f_pred, f_cost, 
				                 prepare_data, test, kf_test)

		history.append([valid_cost, valid_err, test_cost, test_err])

		if (best_p is None or
			     valid_cost <= numpy.array(history)[:,0].min()):
		    best_p = unzip(tparams)
		    bad_counter = 0

		print('%4d: Acc (Cost): Tr %7.5f (%7.5f) Va %7.5f (%7.5f) Te %7.5f (%7.5f)' 
		      % (eidx, 1-train_err, train_cost, 1-valid_err, 
                         valid_cost, 1-test_err, test_cost) )
		print_weights(tparams, model_options)

		if (patience > 0 and len(history) > patience and
                        valid_cost >= numpy.array(history)[:-patience,0].min()):
		    bad_counter += 1
		    if bad_counter > patience:
		        print('Early Stop!')
		        estop = True
		        break

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
    train_err, train_cost = pred_error(f_pred, f_cost,
                                       prepare_data, train, kf_train_sorted)
    valid_err, valid_cost = pred_error(f_pred, f_cost,
                                       prepare_data, valid, kf_valid)
    test_err, test_cost  = pred_error(f_pred, f_cost,
                                      prepare_data, test, kf_test)

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

    print_weights(tparams, model_options)
    return train_err, valid_err, test_err



if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_model(arch_remap_output=True, arch_remap_input=True, n_hpu=100)
