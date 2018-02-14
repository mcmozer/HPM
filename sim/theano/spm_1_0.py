'''
3/26/2017 quickie implementation of Rob Lindsey's SURVIVAL PROCESS MEMORY (SPM)
          as a variant of LSTM with exponential state-dependent forgetting rate
'''
 
from __future__ import print_function
import six.moves.cPickle as pickle

from sklearn import metrics
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

init_wt_mag = 1.0

############################################################################
#theano.config.exception_verbosity = 'high' ################################
#theano.config.compute_test_value = 'raise' ################################
#theano.config.optimizer = 'None'           ################################
#theano.config.mode = 'FAST_COMPILE'        ################################
############################################################################

# Set the random number generators' seeds for consistency
SEED = 123
#numpy.random.seed()  
numpy.random.seed(SEED) #to replicate

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
    Used to restore best parameters
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

    # embedding
    if options['arch_input_map_constraint'] == 'weak':
        # get activations toward saturation with 2.0 factor
        wemb = 2.0 * numpy.concatenate( (numpy.zeros((1,options['n_hid'])),
                                         numpy.identity(options['n_hid'])), axis=0)
        params['Wemb'] = wemb.astype(config.floatX)
    elif options['arch_input_map_constraint'] == 'none':
        # need to add one embedding vector for no-input
        wemb = 0.01 * numpy.random.randn(options['n_in']+1,options['n_hid']);
        params['Wemb'] = wemb.astype(config.floatX)

    # classifier
    if (options['arch_output_fn'] != '1-1'):
       params['U'] = (init_wt_mag / numpy.sqrt(options['n_out']) * 
                                  numpy.random.randn(options['n_hid'],
       				  options['n_out']).astype(config.floatX))
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
# param_init_spm
###############################################################################

def param_init_spm(options, params, prefix='spm'):
    """
    Init the spm parameter:

    :see: init_params
    """
    n_hid = options['n_hid']
    W = numpy.concatenate([ortho_weight(n_hid),
                           ortho_weight(n_hid),
                           ortho_weight(n_hid),
                           ortho_weight(n_hid)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(n_hid),
                           ortho_weight(n_hid),
                           ortho_weight(n_hid),
                           ortho_weight(n_hid)], axis=1)

    params[_p(prefix, 'U')] = U
    # biases are ordered:
    b = numpy.concatenate([0.0 * numpy.ones((n_hid)),            # input gate
                           0.0 * numpy.ones((n_hid)),            # forget gate
                           0.0 * numpy.ones((n_hid)),            # output gate
                           0.0 * numpy.ones((n_hid))], axis=0)   # input
    #b = numpy.zeros((4 * n_hid,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

###############################################################################
# spm_layer
###############################################################################

def spm_layer(tparams, state_below, xt, yt, options, prefix='spm', mask=None):
    # xt and yt are the elapsed time from last input and to next output

    n_hid = options['n_hid']
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
        # h_ has dim n_training_examples  X  n_spm
        # delta_t_input has dim n_training_examples

        # x_ = W_h * x_t + b_h
        preact = x_ + tensor.dot(h_, tparams[_p(prefix, 'U')])

        # input to unit c \equiv (W_h * x_t + U_h * h_{t-1} + b_h) in Rob's notation
        # NOTE: h_{t-1} incorporates decay from output gate last update
        # if it is implemented, then it should have time-dependent properties too

        c = tensor.tanh(_slice(preact, 3, n_hid))

        if (options['arch_spm_include_input_gate']):
            c *= tensor.nnet.sigmoid(_slice(preact, 0, n_hid))

        if (options['arch_spm_include_forget_gate']):
            input_timescale = tensor.exp(_slice(preact, 1, n_hid))
            c += tensor.exp(-delta_t_input[:,None]/input_timescale) * c_ 
        else:
            c += c_

        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        if (options['arch_spm_include_output_gate']):
            output_timescale = tensor.exp(_slice(preact, 2, n_hid))
            h = tensor.exp(-delta_t_output[:,None]/output_timescale) * tensor.tanh(c)
        else:
            h = tensor.tanh(c)

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

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
    yt = tensor.matrix('yt', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_examples = x.shape[1]

    if options['arch_input_map_constraint'] != 'strong':
       emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
						   n_examples,
						   options['n_hid']])
    else:  # weak or none
       Wemb = theano.shared( numpy.concatenate(
                      (numpy.zeros((1,options['n_hid']),dtype=config.floatX),
                       numpy.identity(options['n_hid'],dtype=config.floatX)),
                       axis=0), name='Wemb')
       emb = Wemb[x.flatten()].reshape([n_timesteps,
                                        n_examples,
                                        options['n_hid']])

    if (options['encoder'] == 'spm'):
        proj = get_layer(options['encoder'])[1](
                                            tparams, emb, xt, yt, options,
                                            prefix=options['encoder'],
                                            mask=mask)
        h = c = None

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
            pred_prob_step = (proj_step+1.0e-6) / tensor.sum(proj_step+1.0e-6,axis=1,keepdims=True)
            # No longer needed if there's no '0' output
	    #pred_prob_step = tensor.concatenate([tensor.alloc(0,n_examples,1),
	    #                                     pred_prob_step], axis=1)
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
                             pred_prob_step[tensor.arange(n_examples),y_step-1])
        pred_ix_step = tensor.argmax(pred_prob_step,axis=1) + 1
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
        #tgt_prob_step = tensor.switch(tensor.eq(y_step, 0), 1.0, 
        #             tensor.switch(tensor.gt(y_step, 0),
        #                pred_prob_step[tensor.arange(n_examples),y_step-1],
        #                1.0-pred_prob_step[tensor.arange(n_examples),-y_step-1]))
        tgt_prob_step = tensor.switch(tensor.eq(y_step, 0), 1.0, 
                    tensor.switch(tensor.gt(y_step, 0), 
                                  pred_prob_step[tensor.arange(n_examples), abs(y_step)-1],
                                  1.0-pred_prob_step[tensor.arange(n_examples), abs(y_step)-1]))

        corr_step = tensor.switch(tensor.eq(y_step, 0), 0,
                     tensor.switch(tensor.gt(tgt_prob_step, 0.5), 1, -1))
        return tgt_prob_step, corr_step

    if (options['signed_out']):
        num_alt = 2
        cost_fn = _cost_step_tgt
    else:
        num_alt =  options['n_out']
        cost_fn = _cost_step_norm

    (tgt_prob, corr), updates = theano.scan(cost_fn,
                                            sequences=[pred_prob, y],
                                            outputs_info=None,
                                            non_sequences=None,
                                            n_steps=n_timesteps)
    # New scheme to prevent focus on impossibly bad predictions 1/19/17
    off = 1.0 / numpy_floatX(num_alt) / 1000.
    if tgt_prob.dtype == 'float16':
        off = numpy.max(1e-6, off)
    # tgt_prob: probability correct (dimensions n_timesteps X n_examples)
    cost = -tensor.sum(tensor.log(tgt_prob.clip(off, 1.0))) 
    # Note: not dividing by count because it will reweight minibatch by size
    #         / tensor.sum(tensor.gt(y,0))

    return use_noise, x, xt, y, yt, mask, pred_prob, corr, cost, proj, h, c, tgt_prob


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

def pred_error(f_corr, f_cost, f_pred_prob, prepare_data, data, iterator, options, verbose=False):
    """
    Just compute the error
    f_corr: Theano function to compute the accuracy for a minibatch
    f_cost: Theano function to compute the cost of a minibatch
    prepare_data: usual prepare_data for that dataset.
    """
    correct_ct = all_ct = 0
    cost = 0.0
    predicted = numpy_floatX([])
    target = numpy_floatX([])
    if not options['signed_out']:
        predicted = predicted.reshape(0,options['n_out'])
        target = target.reshape(0,options['n_out'])
    for _, trial_ix in iterator:
        x, xt, y, yt, mask = prepare_data([data[X][t] for t in trial_ix],
	                                  [data[XT][t] for t in trial_ix],
	                                  [data[Y][t] for t in trial_ix],
	                                  [data[YT][t] for t in trial_ix],
                                          maxlen=None)
	cost += f_cost(x, xt, y, yt, mask) 
        corr = f_corr(x, xt, y, yt, mask)
        mask_ravel_bool = (mask.ravel()).astype(bool)
        pred = f_pred_prob(x, xt, yt, mask) # pred_prob dim: n_timesteps x n_examples x n_outputs

        # eliminate don't-care and masked outputs
        y = y.ravel()
        pred = pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2])
        good_rows = ((y!=0) & mask_ravel_bool)
        y = y[good_rows]
        pred = pred[good_rows,:]
        if (options['signed_out']):
            # for signed outputs, only be concerned with the one relevant output
            # pull out target
            pred = pred[numpy.arange(pred.shape[0]),abs(y)-1]
            y = (y > 0.)
        else:
            # for 1-of-N outputs, compute ROC across output classes
            y_for_auc = numpy.zeros((len(y),options['n_out']), dtype=bool)
            y_for_auc[numpy.arange(len(y)),y-1] = True
            y = y_for_auc
        predicted = numpy.concatenate((predicted, pred))
        target = numpy.concatenate((target, y))
        correct_ct += (corr > 0).sum()
        all_ct += (corr != 0).sum()
    # make sure that there's a match between the two ways that masked & 
    # don't-care examples have been removed
    assert(len(target) == all_ct)
    err = 1.0 - numpy_floatX(correct_ct) / numpy_floatX(all_ct)
    cost = cost / numpy_floatX(all_ct)
    auc = metrics.roc_auc_score(target, predicted, average='micro')

    return err, cost, auc

def print_weights(tparams,options):
    print_params = unzip(tparams)
    if (options['arch_output_fn'] != '1-1'):
        print('U:\n',print_params['U'])
        print('b:\n',print_params['b'])
    if options['arch_input_map_constraint'] != 'strong':
        print('Wemb:\n',print_params['Wemb'])

def recode_value_as_index(data):
    new_data = [[abs(data[n][j])*2-(data[n][j]<0) 
                   for j in range(len(data[n]))] for n in range(len(data))]
    return new_data

def train_model(
    arch_input_map_constraint='none', # none: random weights; 'weak': init to identity; 'strong': force to identity
    arch_output_fn='softmax', # 'softmax', 'logistic', or '1-1'
    arch_spm_include_input_gate = True, # spm has input gate 
    arch_spm_include_forget_gate = True, # spm has time-dependent forget gate 
    arch_spm_include_output_gate = True, # spm has time-dependent output gate 
    n_hid=40,  # number of hidden units; ignored if arch_input_map_constraint!='none'
    patience=10,  # Number of epoch to wait before early stop if no progress
                  # -1 = no early stopping
    save_freq=-1,  # Save the parameters after every save_freq updates
                   # -1 = no save
    valid_freq=5,  # Compute the validation error after this number of update.
    max_epochs=1500,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentum and decaying learning rate).
    encoder='spm',  # spm 
    saveto='weights/spm_model.npz',  # The best model will be saved there
    maxlen=500,  # Sequence longer then this get ignored
    batch_size=16, # The batch size during training.
    valid_batch_size=16,  # The batch size used for validation/test set.
    data_file='../data/type_token/hp_4streams_5tokens_1',
    noise_std=0.,
    use_dropout=False,  
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test examples.
    valid_portion=.10, # proportion of training examples for validation
    show_weights = False, # show weights every valid_freq epochs
    type_token_sim = False # sim that lumps together groups of 5 outputs 
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    global layers
    if model_options['encoder'] == 'spm':
        layers = {'spm': (param_init_spm, spm_layer)}
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

    model_options['n_in'] = numpy.max(list(chain(*train[X])))
    model_options['signed_out'] = numpy.min(list(chain(*train[Y]))) < 0
    model_options['n_out'] = numpy.max(numpy.absolute(list(chain(*train[Y])))) 
    if (model_options['signed_out']):
        model_options['arch_output_fn'] = 'logistic'
    if (model_options['arch_output_fn'] != '1-1' and
                            model_options['arch_output_fn'] != 'logistic' and
                            model_options['arch_output_fn'] != 'softmax'):
        print('Error: invalid arch_output_fn')
        quit()

    # if 1-1 mapping between inputs and hidden, then n_hid is forced to 
    # input max
    if (model_options['arch_input_map_constraint'] != 'none' 
                and model_options['arch_output_fn']=='1-1'
                and model_options['n_out'] != model_options['n_in']):
       print('Error: # input and output alternatives must match');
       quit()
    if (model_options['arch_input_map_constraint'] != 'none' 
                and model_options['n_hid'] != model_options['n_in']):
       model_options['n_hid'] = model_options['n_in']
       print('Warning: forcing n_hid = n_in');
    if (model_options['arch_output_fn'] == '1-1'):
       model_options['n_hid'] = model_options['n_out']

    print('Building model with',model_options['n_in'],'inputs,',
          model_options['n_hid'],'hidden, and',model_options['n_out'],
          ['','signed'][model_options['signed_out']],'outputs')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params(saveto, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, xt, y, yt, mask, pred_prob, corr, cost, 
        proj, h, c, tgt_prob) = build_model(tparams, model_options)

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
    f_pred_prob = theano.function([x, xt, yt, mask], pred_prob, name='f_pred_prob')
    # cost function for predicting one-of-N poassible events

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, xt, y, yt, mask], grads, name='f_grad')

    # DEBUG output functions
    #f_h = theano.function([x, xt, yt, mask], h, name='f_h')
    #f_c = theano.function([x, xt, yt, mask], c, name='f_c')
    #f_proj = theano.function([x, xt, yt, mask], proj, name='f_proj')
    #f_tgt_prob = theano.function([x, xt, y, yt, mask], 
    #                             tgt_prob, name='f_tgt_prob')

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
                #tgt_prob = f_tgt_prob(x, xt, y, yt, mask)
                #print('tgt_prob (#steps X # examples X #out?):\n',tgt_prob)


                cost = f_grad_shared(x, xt, y, yt, mask) # compute gradient
                f_update(lrate) # update weights

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
		    print_weights(tparams, model_options)
                    return 1., 1., 1.

            # end minibatch loop

	    if numpy.mod(eidx, valid_freq) == 0:
		use_noise.set_value(0.)
                kf_train_sorted = get_minibatches_idx(len(train[X]), batch_size)
                train_err, train_cost, train_auc = pred_error(
                                      f_corr,f_cost,f_pred_prob,prepare_data,
                                      train,kf_train_sorted,model_options)
		valid_err, valid_cost, valid_auc  = pred_error(
                                      f_corr,f_cost,f_pred_prob,
                                      prepare_data,valid,kf_valid,model_options)
		test_err, test_cost, test_auc  = pred_error(
                                      f_corr,f_cost,f_pred_prob,
                                      prepare_data,test,kf_test,model_options)

		print('%4d: Cost/Acc/AUC: Tr %7.5f %7.5f %7.5f Va %7.5f %7.5f %7.5f Te %7.5f %7.5f %7.5f'
		      % (eidx, train_cost, 1-train_err, train_auc,
                         valid_cost, 1-valid_err, valid_auc,
                         test_cost, 1-test_err, test_auc) )

		if (best_p is None or
			     valid_cost <= numpy.array(history)[:,0].min() or
                             valid_auc >= numpy.array(history)[:,2].max()):
		    best_p = unzip(tparams)
                    best_eidx = eidx

		if (eidx - best_eidx > patience):
                    print('Early Stop!')
                    estop = True
                    break

		history.append([valid_cost, valid_err, valid_auc, 
                                test_cost, test_err, test_auc])

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
        best_eidx = eidx

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[X]), batch_size)

    train_err, train_cost, train_auc = pred_error(
                          f_corr,f_cost,f_pred_prob,prepare_data,
                          train,kf_train_sorted,model_options)
    valid_err, valid_cost, valid_auc  = pred_error(
                          f_corr,f_cost,f_pred_prob,
                          prepare_data,valid,kf_valid,model_options)
    test_err, test_cost, test_auc  = pred_error(
                          f_corr,f_cost,f_pred_prob,
                          prepare_data,test,kf_test,model_options)

    print('%4d: Cost/Acc/AUC: Tr %7.5f %7.5f %7.5f Va %7.5f %7.5f %7.5f Te %7.5f %7.5f %7.5f'
          % (best_eidx, train_cost, 1-train_err, train_auc,
             valid_cost, 1-valid_err, valid_auc,
             test_cost, 1-test_err, test_auc) )
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
    return test_err, test_cost, test_auc
