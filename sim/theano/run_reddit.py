import numpy

# RUN LSTM WITH ADDITIONAL INPUTS
filename = '../data/reddit/reddit'
if 0:
    import hpm_030517
    print('RUNNING LSTM')
    tr, va, te = hpm_030517.train_model(encoder='lstm',
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

# RUN HPM_0416A
if 0:
    import hpm_0416A
    print('RUNNING HPM')
    tr, va, te = hpm_0416A.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False,
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        arch_hpm_gamma_scaled_mu=True,
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

# RUN HPM
if 0:
    import hpm_0102
    tr, va, te = hpm_0102.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none', # NOTE
                        arch_hpm_alpha_constraint='none', # NOTE
                        arch_hpm_recurrent=True, 
                        arch_hpm_prior_exp=False,
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
                        saveto='weights/hpm_0102_reddit.npz',
                        arch_output_fn='softmax')
    print(1-te)

# RUN HPM
if 0:
    import hpm_030517
    tr, va, te = hpm_030517.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none', # NOTE
                        arch_hpm_alpha_constraint='none', # NOTE
                        arch_hpm_recurrent=True, 
                        arch_hpm_prior_exp=False,
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
                        saveto='weights/hpm_030517_reddit.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0:
    import hpm_031117
    tr, va, te = hpm_031117.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_input_map_constraint='none', # NOTE
                        arch_hpm_alpha_exp=0.0,
                        arch_hpm_mu_exp=0.1, 
                        arch_hpm_prior_exp=0.9,
                        arch_hpm_recurrent=True, 
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
                        saveto='weights/hpm_031117_reddit.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0:
    import hpm_0417A
    filename = '../data/reddit/reddit'
    tr, va, te = hpm_0417A.train_model(encoder='hpm',
                        arch_hpm_gated=False, 
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        arch_hpm_gamma_scaled_mu=True, 
                        arch_hpm_gamma_scaled_alpha=True, 
                        arch_hpm_prior_exp=True, 
                        timescales=2.0**numpy.arange(-7,7),
                        # Note: arange(-7,7) = -7:6
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=25,
                        n_hid=50, 
                        data_file=filename,
                        arch_output_fn='softmax') 
    print(1-te)

if 0:
    import hpm_1104
    tr, va, te = hpm_1104.train_model(encoder='hpm',
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
    import hpm_1105
    tr, va, te = hpm_1105.train_model(encoder='hpm',
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
                        saveto='weights/hpm_model_1105.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0:
    import hpm_1110
    tr, va, te = hpm_1110.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False,
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        arch_hpm_gamma_scaled_mu=False, # should be False
                        arch_hpm_gamma_scaled_alpha=False, # should be False
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
                        saveto='weights/hpm_1110.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0:
    import hpm_1113
    tr, va, te = hpm_1113.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=False,
                        arch_remap_input=True,
                        arch_hpm_recurrent=True, 
                        arch_hpm_gamma_scaled_mu=False, # should be False
                        arch_hpm_gamma_scaled_alpha=False, # should be False
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
                        saveto='weights/hpm_1110.npz',
                        arch_output_fn='softmax')
    print(1-te)

if 0: # RUN SPM 
    import spm 
    tr, va, te = spm.train_model(encoder='spm',
                    arch_spm_init_forget_timescale=64.0,
                    arch_spm_init_input_timescale=10./60./24.,
                    #arch_spm_include_input_gate=True,
                    #arch_spm_include_forgetting=64.0,
                    #arch_spm_input_includes_gamma=True, # DEBUG
                    valid_portion=.15,
                    valid_freq=1,
                    maxlen=1000,
                    patience=25,
                    n_hid=50,
                    data_file=filename,
                    saveto='weights/spm_model.npz',
                    arch_output_fn='softmax')


if 1: # RUN CTGRU 
    import gru_5_2 as gru 
    print "RUNNING CTGRU 5.2"
    tr, va, te = gru.train_model(encoder='ctgru',
                    arch_ctgru_ohat_for_sscale=False, ###########
                    arch_ctgru_include_priors=False,  ###########
                    #arch_ctgru_include_delta_t=False, ###########
                    timescales=10.0**numpy.arange(-2.1,2.6,.5),
                    valid_portion=.15,
                    valid_freq=1,
                    maxlen=1000,
                    patience=25,
                    n_hid=50,
                    data_file=filename,
                    saveto='weights/ctgru_reddit_model.npz',
                    arch_output_fn='softmax')

if 0:
    import gru_5_2 as gru 
    print "RUNNING GRU 5.2"
    tr, va, te = gru.train_model(encoder='gru',
                    arch_gru_include_delta_t=True,
                    valid_portion=.15,
                    valid_freq=1,
                    maxlen=1000,
                    patience=25,
                    n_hid=50,
                    data_file=filename,
                    saveto='weights/spm_model.npz',
                    arch_output_fn='softmax')


