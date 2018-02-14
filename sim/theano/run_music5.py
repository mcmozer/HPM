import numpy
import hpm_0102

# RUN LSTM
if 1:
    te, tll, tauc = hpm_0102.train_model(encoder='lstm',
                arch_lstm_include_delta_t=True,
                arch_lstm_include_input_gate=True,
                arch_lstm_include_forget_gate=True,
                arch_lstm_include_output_gate=True,
                valid_portion=.15,
                data_file='../data/synthetic_music/5streams',
                valid_freq=5,
                patience = 100,
                n_hid=5,
                saveto='weights/lstm_music.npz',
                arch_output_fn='softmax')

# RUN HPM
if 0:
    te, tll, tauc = hpm_0102.train_model(encoder='hpm',
                valid_portion=.15,
                show_weights=False,
                arch_hpm_gated=True,
                arch_remap_input=True,
                arch_hpm_recurrent=True, 
                #arch_hpm_gamma_scaled_mu=True,
                #arch_hpm_gamma_scaled_alpha=True,
                arch_hpm_prior_exp=False,
                timescales=2.0**numpy.arange(0,11),
                data_file='../data/synthetic_music/5streams',
                valid_freq=5,
                patience=100,
                n_hid=5,
                saveto='weights/hpm_music.npz',
                arch_output_fn='softmax')

print('acc ',1-te, ' logL ',tll, ' auc ',tauc)

