import numpy

# RUN LSTM
if (0):
    te, tll, tauc = hpm_0102.train_model(encoder='lstm',
                arch_lstm_include_delta_t=True,
                arch_lstm_include_input_gate=True,
                arch_lstm_include_forget_gate=True,
                arch_lstm_include_output_gate=True,
                valid_portion=.15,
                valid_freq=5,
                patience = 100,
                n_hid=25,
                saveto='weights/lstm_msnbc.npz',
                arch_output_fn='softmax')

# RUN HPM 0102
if 0:
    import hpm_0102
    te, tll, tauc = hpm_0102.train_model(encoder='hpm',
                valid_portion=.15,
                show_weights=False,
                arch_hpm_gated=True,
                arch_remap_input=True,
                arch_hpm_recurrent=True, 
                arch_hpm_prior_exp=False,
                timescales=2.0**numpy.arange(0,7),
                data_file='../data/msnbc/msnbc',
                valid_freq=5,
                patience=100,
                n_hid=25,
                saveto='weights/hpm_msnbc.npz',
                arch_output_fn='softmax')

# RUN HPM 031117
if 0:
    import hpm_031117 # version with mixture of time scales
    te, tll, tauc = hpm_031117.train_model(encoder='hpm',
                show_weights=False,
                arch_hpm_gated=True,
                arch_input_map_constraint='none',
                arch_hpm_alpha_exp=0.0,
                arch_hpm_mu_exp=0.1,
                arch_hpm_prior_exp=0.9,
                arch_hpm_recurrent=True,
                timescales=2.0**numpy.arange(0,7),
                data_file='../data/msnbc/msnbc',
                valid_freq=5,
                patience=25,
                n_hid=25,
                valid_portion=.15,
                saveto='weights/hpm_031117_msnbc.npz',
                arch_output_fn='softmax')
if 0:
    import spm
    te, tll, tauc = spm.train_model(encoder='spm',
                show_weights=False,
                arch_spm_init_forget_timescale=50.,
                arch_spm_init_input_timescale=1.,
                data_file='../data/msnbc/msnbc',
                valid_freq=5,
                patience=25,
                n_hid=25,
                valid_portion=.15,
                saveto='weights/spm_msnbc.npz',
                arch_output_fn='softmax')


if 1:
    import gru_5_2 as gru
    for h in [50]: # [25, 50]:
        print "RUNNING GRU WITH " , h , " HID"
        te,tll,tauc = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False,
                        arch_ctgru_include_priors=False,
                        arch_gru_include_delta_t = False, # no need for msnbc
                        timescales=10**numpy.arange(0.0,2.5,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25, 
                        n_hid=h, 
                        data_file='../data/msnbc/msnbc',
                        saveto='weights/gru_msnbc.npz',
                        arch_output_fn='softmax')
        print('acc ',1-te, ' logL ',tll, ' auc ',tauc)

if 0:
    import gru_5_2 as gru
    for h in [25, 50]:
        print "RUNNING CTGRU WITH " , h , " HID"
        te,tll,tauc = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False,
                        arch_ctgru_include_priors=False,
                        #arch_gru_include_delta_t = True,
                        timescales=10**numpy.arange(0.0,2.5,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25, 
                        n_hid=h, 
                        data_file='../data/msnbc/msnbc',
                        saveto='weights/ctgru_msnbc.npz',
                        arch_output_fn='softmax')
        print('acc ',1-te, ' logL ',tll, ' auc ',tauc)
