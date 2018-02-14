import numpy
import hpm_onescale

# RUN LSTM
if 1:
    tr, va, te = hpm_onescale.train_model(encoder='lstm',
                data_file='../data/exeqrep/data',
                valid_freq=1,
                arch_lstm_include_delta_t=True, 
                n_hid=25,
                arch_output_fn='logistic')
# RUN HPM
if 0:
    tr, va, te = hpm_onescale.train_model(encoder='hpm',
                data_file='../data/exeqrep/data',
                valid_freq=1,
                patience=100,
                arch_output_fn='logistic',
                n_hid=25,
                arch_hpm_recurrent=True)
