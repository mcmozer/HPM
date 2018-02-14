import numpy
import hpm

# original one-shot forgetting
# RUN LSTM
#if 1:
#    tr, va, te = hpm.train_model(encoder='lstm',
#                data_file='../data/forgetting/forgetting0',
#                valid_freq=1,
#                arch_lstm_include_delta_t=True, 
#                patience=50,
#                n_hid=25,
#                arch_output_fn='logistic')
## RUN HPM
#if 0:
#    tr, va, te = hpm.train_model(encoder='hpm',
#                data_file='../data/forgetting/forgetting0',
#                valid_freq=1,
#                patience=50,
#                timescales=[1.,2.,4.,8.,16.,32.,64.],
#                #timescales=2.0**numpy.arange(-11,7),
#                arch_output_fn='logistic',
#                n_hid=25,
#                arch_remap_input=True,
#                arch_hpm_recurrent=True)


filename = '../data/forgetting/forgetting10'
# 10 event streams
# inter-event times of 1, 10, or 100
# next occurrence of event will be + if within 310 steps, - otherwise


if 1: 
    import gru_5_3 as gru
    for n_hid in [10, 20, 40]:
        print "RUNNING GRU WITH ", n_hid, " HIDDEN"
        tr,va,te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=False, # DEBUG *******
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25, 
                        n_hid=n_hid,
                        data_file=filename,
                        saveto='weights/gru_hp_10streams.npz',
                        arch_output_fn='logistic')

if 1: 
    import gru_5_2 as gru
    for n_hid in [10, 20, 40]:
        print "RUNNING CTGRU 5.1 WITH ", n_hid, " HIDDEN"
        tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_include_priors=False, 
                        arch_ctgru_include_ohat_for_sscale=False,
                        timescales=10**numpy.arange(0.0,3.5,.5), # **************
                        #timescales=[1.0, 10., 100., 1000.], # DEBUG
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=n_hid,
                        data_file=filename,
                        saveto='weights/gru53_forgetting.npz',
                        arch_output_fn='logistic')

