import numpy
import hpm_gated

# RUN LSTM
if 0:
    tr, va, te = hpm_gated.train_model(encoder='lstm',
                data_file='../data/exeqrep/data',
                valid_freq=1,
                arch_lstm_include_delta_t=True, 
                n_hid=25,
                arch_output_fn='logistic')
# RUN HPM
if 1:
    tr, va, te = hpm_gated.train_model(encoder='hpm',
                data_file='../data/exeqrep/data',
                valid_freq=1,
                patience=100,
                arch_remap_input=True,
                arch_output_fn='logistic',
                timescales=[.01,.1,1,10,100],
                optimizer=hpm_gated.adadelta,
                #show_weights=True,
                n_hid=25,
                arch_hpm_recurrent=True) 
