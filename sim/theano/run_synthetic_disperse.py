import numpy

filename = '../data/synthetic_disperse/disperse2'
if 0: 
    import gru_5_2 as gru
    print "RUNNING GRU WITH DELTA T"
    tr,va,te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=True,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_disperse.npz',
                        arch_output_fn='logistic')
if 1: 
    import gru_5_2 as gru
    print "RUNNING CTGRU 5.2"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False, # DEBUG
                        arch_ctgru_include_priors=False, # DEBUG
                        timescales=10**numpy.arange(0,2.5,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=40, # DEBUG
                        data_file=filename,
                        saveto='weights/hpm_synthetic_disperse.npz',
                        arch_output_fn='logistic')
if 0: 
    import gru_5_2_test as gru
    print "RUNNING CTGRU 5.2 TEST -- NO DECAY"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False, # DEBUG
                        arch_ctgru_include_priors=False, # DEBUG
                        timescales=10**numpy.arange(0,2.5,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_disperse.npz',
                        arch_output_fn='logistic')

if 0:
    import gru_5_2 as gru
    print "RUNNING GRU WITHOUT DELTA T"
    tr,va,te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=False,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_disperse.npz',
                        arch_output_fn='logistic')

