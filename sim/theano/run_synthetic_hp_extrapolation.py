import numpy

filename = '../data/synthetic_hp_extrapolation/hp_10streams'

# RUN LSTM WITH ADDITIONAL INPUTS
if 1:
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

    for ifv in [1000]: # [4000, 2000.,1000.,100.,1.]:
        tr, va, te = spm.train_model(encoder='spm',
                            show_weights=False,
                            arch_input_map_constraint='none',
                            arch_spm_include_input_gate=True,
                            arch_spm_include_forgetting=ifv, 
                            arch_spm_input_includes_gamma=False, 
                            valid_portion=.15,
                            valid_freq=1,
                            maxlen=1000,
                            patience=25,
                            n_hid=20, 
                            data_file=filename,
                            saveto='weights/spm_hp_10streams.npz',
                            arch_output_fn='softmax')
        print(1-te)
