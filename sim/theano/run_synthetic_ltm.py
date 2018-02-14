import numpy

filename = '../data/synthetic_ltm/ltm'
if 1: 
    import gru_5_2 as gru
    print "RUNNING CTGRU 5.2"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False, # DEBUG
                        arch_ctgru_include_priors=False, # DEBUG
                        timescales=10**numpy.arange(0,3.0,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=30, # WAS 15,
                        data_file=filename,
                        saveto='weights/ctgru_synthetic_ltm.npz',
                        arch_output_fn='logistic')

if 1: 
    import gru_5_2 as gru
    print "RUNNING GRU WITH DELTA T"
    tr,va,te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=True,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15, 
                        n_hid=30, # WAS 15,
                        data_file=filename,
                        saveto='weights/gru_synthetic_ltm.npz',
                        arch_output_fn='logistic')
