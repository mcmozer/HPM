import numpy
import hpm_0102_test

te, tll, tauc = hpm_0102_test.train_model(
                valid_portion=.15,
                valid_freq=1,
                data_file='../data/reddit/reddit',
                show_weights=False,
                arch_include_delta_t=True,
                arch_lstm_include_input_gate=True,
                arch_lstm_include_forget_gate=True,
                arch_lstm_include_output_gate=True,
                arch_hpm_gated=True,
                arch_hpm_prior_exp=False,
                timescales=2.0**numpy.arange(-7,7),
                    # NOTE: arange(-7,7) = -7:6
                patience=25,
                maxlen=1000,
                hpm_n_hid=25,
                lstm_n_hid=25,
                saveto='weights/lstm_and_hpm_reddit.npz',
                arch_output_fn='softmax')

print('acc ',1-te, ' logL ',tll, ' auc ',tauc)

