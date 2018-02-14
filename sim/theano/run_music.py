import numpy
import hpm_1226

# RUN LSTM
if 1:
    tr, va, te = hpm_1226.train_model(encoder='lstm',
                arch_lstm_include_delta_t=True,
                valid_portion=.15,
                data_file='../data/synthetic_music/40streams',
                valid_freq=1,
                patience = 20,
                n_hid=40,
                saveto='weights/lstm_music.npz',
                arch_output_fn='softmax')

# RUN HPM
if 1:
    tr, va, te = hpm_1226.train_model(encoder='hpm',
                valid_portion=.15,
                show_weights=False,
                arch_hpm_gated=False,
                arch_remap_input=True,
                arch_hpm_recurrent=True, 
                arch_hpm_gamma_scaled_mu=True,
                arch_hpm_prior_exp=True,
                timescales=2.0**numpy.arange(0,11),
                data_file='../data/synthetic_music/40streams',
                valid_freq=1,
                patience=20,
                n_hid=40,
                saveto='weights/hpm_1226_music.npz',
                arch_output_fn='softmax')
    print(1-te)

