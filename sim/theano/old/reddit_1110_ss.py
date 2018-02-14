import numpy
import hpm_1110_ss

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/reddit/reddit'
if 0:
    tr, va, te = hpm_1110_ss.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50,
                        data_file=filename,
                        saveto='lstm_model.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 1:
    tr, va, te = hpm_1110_ss.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False,
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50, 
                        data_file=filename,
                        saveto='hpm_1110_ss.npz',
                        arch_output_fn='softmax')
    print(1-te)

