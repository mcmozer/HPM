import numpy

filename = '../data/synthetic_rhythm/rhythm_4_25'
if 1: 
    import gru_5_3 as gru
    print "RUNNING CTGRU 5.3 with 40 hidden"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_include_priors=False, 
                        arch_ctgru_include_smoothc=False,
                        arch_ctgru_include_delta_t=False,
                        timescales=numpy.array([1.0, 4.0, 16.0, 64.0]),
                        #timescales=numpy.array([1.0, 10.0, 100.]),
                        #timescales=numpy.array([1.0, 2., 4., 8., 16., 32., 64., 128.]),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=40,
                        data_file=filename,
                        saveto='weights/gru53_synthetic_rhythm.npz',
                        arch_output_fn='logistic')

if 0: 
    import gru_5_3 as gru
    print "RUNNING GRU WITH 40 hidden"
    tr,va,te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=True,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=40,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
