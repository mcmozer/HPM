import numpy

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/last_fm/lastfm15k'
if 0:
    import hpm_1110
    tr, va, te = hpm_1110.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50,
                        data_file=filename,
                        saveto='weights/lstm_model.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 0:
    import hpm_1110
    tr, va, te = hpm_1110.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False,
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        arch_hpm_gamma_scaled_mu=True,
                        arch_hpm_gamma_scaled_alpha=True,
                        arch_hpm_prior_exp=True,
                        timescales=2.0**numpy.arange(-7,7),
                        # NOTE: arange(-7,7) = -7:6
                        #timescales=[1./60./60./24., 1./60./24., 1./24., 1., 30.],
                        #timescales=10.0**numpy.arange(-4,3),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50, 
                        data_file=filename,
                        saveto='weights/hpm_model_true.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0:
    import spm
    tr, va, te = spm.train_model(encoder='spm',
                        show_weights=False,
                        arch_spm_include_input_gate=True,
                        arch_spm_include_forgetting=64.0,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50, 
                        data_file=filename,
                        saveto='weights/spm_model_true.npz',
                        arch_output_fn='softmax')
    print(1-te)


if 0:
    import gru_5_2 as gru
    for h in [50]:
        print "RUNNING GRU WITH " , h , " HID"
        te,tll,tauc = gru.train_model(encoder='gru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False,
                        arch_ctgru_include_priors=False,
                        arch_gru_include_delta_t = True,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25, 
                        n_hid=h, 
                        data_file=filename,
                        saveto='weights/gru_lastfm.npz',
                        arch_output_fn='softmax')
        print('acc ',1-te, ' logL ',tll, ' auc ',tauc)

if 1:
    import gru_5_2 as gru
    for h in [50]:
        print "RUNNING CTGRU WITH " , h , " HID"
        te,tll,tauc = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False,
                        arch_ctgru_include_priors=False,
                        #arch_gru_include_delta_t = True,
                        timescales=10**numpy.arange(-2.1,2.4,0.5), # -2.1, -1.6, ... 1.9
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25, 
                        n_hid=h, 
                        data_file=filename,
                        saveto='weights/ctgru_lastfm.npz',
                        arch_output_fn='softmax')
        print('acc ',1-te, ' logL ',tll, ' auc ',tauc)
