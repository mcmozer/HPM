import numpy
import hpm_0416A_debug

# RUN LSTM WITH ADDITIONAL INPUTS
if 0:
    filename = '../data/reddit/reddit'
    tr, va, te = hpm_0416A_debug.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50,
                        data_file=filename,
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 1:
    filename = '../data/reddit/reddit'
    tr, va, te = hpm_0416A_debug.train_model(encoder='hpm',
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        timescales=2.0**numpy.arange(-4,9), # NOTE
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50, 
                        data_file=filename,
                        arch_hpm_gamma_scaled_mu=True, # NOTE
                        arch_hpm_gamma_scaled_alpha=True, # NOTE
                        arch_output_fn='softmax')
    print(1-te)

