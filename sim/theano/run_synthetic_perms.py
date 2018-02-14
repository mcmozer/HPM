import numpy

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/synthetic_perms/synthetic_perms'
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
                        saveto='weights/lstm_perms.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 0:
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_alpha_constraint='none',
                        arch_hpm_recurrent=True,
                        arch_hpm_prior_exp=False,
                        timescales=[1., 10.,  100., 1000.],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=10, 
                        data_file=filename,
                        saveto='weights/hpm_030517_perms.npz',
                        arch_output_fn='softmax')
    print(1-te)

# HPM with no input remapping
if 1:
    import hpm_031117
    tr, va, te = hpm_031117.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False, # CONSTRAINT
                        arch_input_map_constraint='strong', # CONSTRAINT
                        arch_hpm_alpha_exp=0.0, # CONSTRAINT
                        arch_hpm_mu_exp=0.0, # CONSTRAINT
                        arch_hpm_prior_exp=1.0, # CONSTRAINT
                        arch_hpm_recurrent=False, # CONSTRAINT
                        timescales=[1., 10., 100., 1000.],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_031117_perms.npz',
                        arch_output_fn='1-1') # CONSTRAINT
    print(1-te)

