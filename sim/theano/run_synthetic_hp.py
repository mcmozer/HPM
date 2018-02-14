import numpy

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/synthetic_hp/hp_10streams'
if 0:
    import hpm_0102
    tr, va, te = hpm_0102.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN gated multiscale memory (041317)
if 0:
    print "running hpm_041317"
    import hpm_041317 as hpm
    tr, va, te = hpm.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_recurrent=True, 
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_041317_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 0:
    import hpm_0102
    tr, va, te = hpm_0102.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, # DEBUG ***********
                        arch_input_map_constraint='none',
                        arch_hpm_recurrent=True, 
                        arch_hpm_prior_exp=False,
                        arch_hpm_alpha_constraint='strong', # CHEAT!
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_0102_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

# HPM with no input remapping
if 0:
    import hpm_0102
    tr, va, te = hpm_0102.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False, # CHANGED
                        arch_input_map_constraint='strong', # CHANGED
                        arch_hpm_recurrent=False, # CHANGED
                        arch_hpm_prior_exp=False,
                        arch_hpm_alpha_constraint='strong',
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_0102_hp_10streams.npz',
                        arch_output_fn='1-1')
    print(1-te)

# HPM 030517
if 0: 
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_alpha_constraint='weak', # WILL THIS WORK?
                        arch_hpm_recurrent=True,
                        arch_hpm_prior_exp=False,
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_030517_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

# HPM 030517 with no input remapping
if 0:
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False, # CONSTRAINT
                        arch_input_map_constraint='strong', # CONSTRAINT
                        arch_hpm_alpha_constraint='strong',
                        arch_hpm_recurrent=False, # CONSTRAINT
                        arch_hpm_prior_exp=False,
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_030517_hp_10streams.npz',
                        arch_output_fn='1-1') # CONSTRAINT
    print(1-te)

import hpm_031117
# HPM 031117
if 0:
    import hpm_031117
    tr, va, te = hpm_031117.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_alpha_exp=0.8,
                        arch_hpm_mu_exp=0.05, 
                        arch_hpm_prior_exp=0.05,
                        arch_hpm_recurrent=True,
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_031117_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

# HPM 031117 no input remapping
if 0:
    import hpm_031117
    tr, va, te = hpm_031117.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False, # CONSTRAINT
                        arch_input_map_constraint='strong', # CONSTRAINT
                        arch_hpm_alpha_exp=0.0, # CONSTRAINT
                        arch_hpm_mu_exp=0.0, # CONSTRAINT
                        arch_hpm_prior_exp=1.0, # CONSTRAINT
                        arch_hpm_recurrent=False, # CONSTRAINT
                        timescales=2.0**numpy.arange(0,13),
                        # note: does not include 2^13
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_031117_hp_10streams.npz',
                        arch_output_fn='1-1') # CONSTRAINT
    print(1-te)

# SPM (survival process memory)
if 0: 
    import spm

    tr, va, te = spm.train_model(encoder='spm',
                        show_weights=False,
                        arch_input_map_constraint='none',
                        #arch_spm_include_input_gate=True,     v. 2.*
                        #arch_spm_include_forgetting=ifv,      v. 2.*
                        #arch_spm_input_includes_gamma=False,  v. 2.3
                        arch_spm_init_input_timescale=1,
                        arch_spm_init_forget_timescale=100.,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/spm_hp_10streams.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 1: 
    import gru_5_1 as gru
    filename = '../data/synthetic_hp/hp_10streams_10k'
    for n_hid in [80]:
        print "RUNNING CTGRU 5.1 WITH LARGER DATA SET AND",n_hid,"HIDDEN"
        tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        #arch_ctgru_include_priors=False, 
                        #arch_ctgru_include_smoothc=False,
                        #arch_ctgru_include_delta_t=True, # DEBUG
                        #arch_ctgru_include_nodecay_out=True, # DEBUG
                        timescales=10**numpy.arange(0,3.5,.5), # WAS [1.0, 10., 100., 1000.], 
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15,  
                        n_hid=n_hid,
                        data_file=filename,
                        saveto='weights/gru51_hp_10streams.npz',
                        arch_output_fn='softmax')

if 0: 
    import gru_5_3_test as gru
    filename = '../data/synthetic_hp/hp_10streams_10k'
    for n_hid in [80,40]: # [10, 20, 40]:
        print "RUNNING **TEST** CTGRU 5.3 WITH LARGER DATA SET AND",n_hid,"HIDDEN"
        tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_include_priors=False, 
                        arch_ctgru_include_smoothc=False,
                        arch_ctgru_include_delta_t=True, # DEBUG
                        arch_ctgru_include_nodecay_out=True, # DEBUG
                        timescales=[1.0, 10., 100., 1000.],  #***** NOTE BAD
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=15,  
                        n_hid=n_hid,
                        data_file=filename,
                        saveto='weights/gru53_hp_10streams.npz',
                        arch_output_fn='softmax')


if 0: 
    import gru_5_3 as gru
    filename = '../data/synthetic_hp/hp_10streams_10k'
    for n_hid in [80]: #[10, 20, 40]:
        print "RUNNING GRU WITH LARGER DATA SET AND",n_hid,"HIDDEN"
        tr,va,te = gru.train_model(encoder='gru',
                            show_weights=False,
                            arch_gru_include_delta_t=True,
                            valid_portion=.15,
                            valid_freq=1,
                            maxlen=1000,
                            patience=15,
                            n_hid=n_hid,
                            data_file=filename,
                            saveto='weights/gru_hp_10streams.npz',
                            arch_output_fn='softmax')

