import numpy
import hpm_030517

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/synthetic_one_interest/one_interest'
if 1:
    tr, va, te = hpm_030517.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_one_interest.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 1:
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none',
                        arch_hpm_alpha_constraint='none', 
                        arch_hpm_recurrent=True,
                        arch_hpm_prior_exp=False,
                        timescales=2.0**numpy.arange(0,7),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_030517_one_interest.npz',
                        arch_output_fn='softmax')
    print(1-te)
# HPM with no input remapping
if 1:
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False, # CONSTRAINT
                        arch_input_map_constraint='strong', # CONSTRAINT
                        arch_hpm_alpha_constraint='strong',
                        arch_hpm_recurrent=False, # CONSTRAINT
                        arch_hpm_prior_exp=False,
                        timescales=2.0**numpy.arange(0,7),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/hpm_030517_one_interest.npz',
                        arch_output_fn='1-1') # CONSTRAINT
    print(1-te)

