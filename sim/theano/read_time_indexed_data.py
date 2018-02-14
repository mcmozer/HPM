from __future__ import print_function
from six.moves import xrange

import gzip
import os

import numpy
import theano

def prepare_data(ox, oxt, oy, oyt, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    lengths = [len(s) for s in ox]
    if maxlen is not None:
        new_lengths = []
        new_ox = []
        new_oxt = []
        new_oy = []
        new_oyt = []
        for l, lox, loxt, loy, loyt in zip(lengths, ox, oxt, oy, oyt):
            if l < maxlen:
                new_lengths.append(l)
	        new_ox.append(lox)
	        new_oxt.append(loxt)
	        new_oy.append(loy)
	        new_oyt.append(loyt)
        lengths = new_lengths
	ox = new_ox
	oxt = new_oxt
	oy = new_oy
	oyt = new_oyt

        if len(lengths) < 1:
            return None, None, None, None, None

    n_examples = len(ox)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_examples)).astype('int64')
    xt = numpy.zeros((maxlen, n_examples)).astype(theano.config.floatX)
    y = numpy.zeros((maxlen, n_examples)).astype('int64')
    yt = numpy.zeros((maxlen, n_examples)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, n_examples)).astype(theano.config.floatX)
    for idx, s in enumerate(ox):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
    for idx, s in enumerate(oxt):
        xt[:lengths[idx], idx] = s
    for idx, s in enumerate(oy):
        y[:lengths[idx], idx] = s
    for idx, s in enumerate(oyt):
        yt[:lengths[idx], idx] = s

    return x, xt, y, yt, x_mask


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            ".",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

        
    return dataset


def read_time_indexed_data(fname,maxlen=None):
    ievent = []
    itime = []
    oevent = []
    otime = []
    with open(fname) as f:
        dtype = 0
	for l in f:
	    # strip off example #
	    if dtype == 0:
	        cur = [int(x) for x in l.split()]
                ievent.append(cur[1:]) # input events
	    elif dtype == 1:
	        cur = [float(x) for x in l.split()]
	        itime.append(cur[1:]) # input event times
            elif dtype == 2:
	        cur = [int(x) for x in l.split()]
                oevent.append(cur[1:]) # output events
	    else:
	        cur = [float(x) for x in l.split()]
                otime.append(cur[1:]) # output event times
            dtype = (dtype+1) % 4

    # discard the sequences that are too long
    if maxlen:
        new_ievent = []
        new_itime = []
        new_oevent = []
        new_otime = []
        for x, xt, y, yt in zip(ievent, itime, oevent, otime):
            if len(x) < maxlen:
                new_ievent.append(x)
                new_itime.append(xt)
                new_oevent.append(y)
                new_otime.append(yt)
        ievent = new_ievent
        itime = new_itime
        oevent = new_oevent
        otime = new_otime
        del new_ievent, new_itime, new_oevent, new_otime

    print(len(ievent), "examples read from",fname) 
    return ievent, itime, oevent, otime

def load_data(data_file, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    train_set = read_time_indexed_data(data_file + '.train', maxlen)
    test_set = read_time_indexed_data(data_file + '.test', maxlen)

    train_set_x, train_set_xt, train_set_y, train_set_yt = train_set
    test_set_x, test_set_xt, test_set_y, test_set_yt = test_set

    # split training set into validation set
    n_examples = len(train_set_x)
    sidx = numpy.random.permutation(n_examples)
    n_train = int(numpy.round(n_examples * (1. - valid_portion)))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_xt = [train_set_xt[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_yt = [train_set_yt[s] for s in sidx[n_train:]]

    # need to create valid_set first since the ops below affect train_set
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_xt = [train_set_xt[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_yt = [train_set_yt[s] for s in sidx[:n_train]]

    if sort_by_len:
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_xt = [test_set_xt[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        test_set_yt = [test_set_yt[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_xt = [valid_set_xt[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        valid_set_yt = [valid_set_yt[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_xt = [train_set_xt[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        train_set_yt = [train_set_yt[i] for i in sorted_index]

    train_set = (train_set_x, train_set_xt, train_set_y, train_set_yt)
    valid_set = (valid_set_x, valid_set_xt, valid_set_y, valid_set_yt)
    test_set  = (test_set_x,  test_set_xt,  test_set_y,  test_set_yt)

    return train_set, valid_set, test_set
