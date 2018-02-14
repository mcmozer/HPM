import numpy

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/colt/COLT'
if 0:
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=10,
                        data_file=filename,
                        saveto='weights/lstm_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# RUN HPM 030517
if 0:
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none', # NOTE
                        arch_hpm_alpha_constraint='weak', # NOTE
                        arch_hpm_recurrent=True, 
                        arch_hpm_prior_exp=False,
                        timescales=2.0**numpy.arange(-5.0,7.0,2.0),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=10, 
                        data_file=filename,
                        saveto='weights/hpm_030517_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# RUN HPM 031117
if 0:
    import hpm_031117
    tr, va, te = hpm_031117.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_alpha_exp=0.0,
                        arch_hpm_mu_exp=0.1,
                        arch_hpm_prior_exp=0.9,
                        arch_hpm_recurrent=True,
                        timescales=2.0**numpy.arange(-5.0,7.0,2.0),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=10, 
                        data_file=filename,
                        saveto='weights/hpm_031117_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# RUN SPM
if 0:
    import spm
    tr, va, te = spm.train_model(encoder='spm',
                        show_weights=False,
                        arch_spm_init_forget_timescale=120.0,
                        arch_spm_init_input_timescale=0.1,
                        #arch_spm_include_input_gate=True,
                        #arch_spm_include_forgetting=120.0, # numeric value = initial time scale
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=10, 
                        data_file=filename,
                        saveto='weights/spm_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# RUN CTGRU 5.3
if 0:
    import gru_5_3 as gru
    print "RUNNING GRU_5_3"
    for n_hid in [10, 40]:
        tr, va, te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        #arch_ctgru_ohat_for_sscale=True, # version 5.2
                        arch_ctgru_include_priors=False,
                        arch_ctgru_include_smoothc=False,
                        arch_ctgru_include_delta_t=False,
                        timescales=10**numpy.arange(-1.5,2.0,.5),
                        #timescales=[0.0006944,0.0107583,0.1666667,2.5819889,40.0000000],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=n_hid, 
                        data_file=filename,
                        saveto='weights/spm_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# RUN CTGRU 5.1
if 1:
    import gru_5_1 as gru
    print "RUNNING GRU_5_1"
    for n_hid in [10, 40]:
        tr, va, te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=True, # version 5.1
                        #arch_ctgru_include_priors=False,
                        #arch_ctgru_include_smoothc=False,
                        #arch_ctgru_include_delta_t=False,
                        timescales=10**numpy.arange(-1.5,2.0,.5),
                        #timescales=[0.0006944,0.0107583,0.1666667,2.5819889,40.0000000],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=n_hid, 
                        data_file=filename,
                        saveto='weights/spm_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

# REGULAR GRU
if 1:
    import gru_5_3 as gru
    print "RUNNING GRU via 5.2 WITH DELTA T"
    for n_hid in [10, 40]:
        tr, va, te = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_gru_include_delta_t=True,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=n_hid, 
                        data_file=filename,
                        saveto='weights/spm_colt.npz',
                        arch_output_fn='logistic')
    print(1-te)

