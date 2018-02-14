import numpy

filename = '../data/synthetic_cluster/cluster3'
if 0: 
    import gru_5_3 as gru
    print "RUNNING CTGRU 5.3 WITH NO PRIORS, SMOOTHC, DELTAT -- 10 hidden"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_include_priors=False, # DEBUG
                        arch_ctgru_include_smoothc=False, # DEBUG
                        arch_ctgru_include_delta_t=False, # DEBUG
                        timescales=[0.1, 1.0, 10.0, 100.0],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=10,  # DEBUG
                        data_file=filename,
                        saveto='weights/gru53_synthetic_cluster.npz',
                        arch_output_fn='logistic')

if 0: 
    import gru_5_2 as gru
    print "RUNNING CTGRU 5.2"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        arch_ctgru_ohat_for_sscale=False, # DEBUG
                        arch_ctgru_include_priors=False, # DEBUG
                        timescales=10**numpy.arange(-1.0,3.0,.5),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
if 0: 
    import gru_5_0 as gru
    print "RUNNING CTGRU 5.0"
    tr,va,te = gru.train_model(encoder='ctgru',
                        show_weights=False,
                        timescales=[0.1, 1.0, 10.0, 100.0],
                        arch_ctgru_include_delta_t=False,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
if 0: 
    import hpm_041317 as hpm
    print "RUNNING HPM 041317"
    tr,va,te = hpm.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True,  
                        arch_hpm_recurrent=True,
                        timescales=[0.1, 1.0, 10.0, 100.0],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
if 0: 
    import hpm_041417 as hpm
    print "RUNNING HPM 041417"
    tr,va,te = hpm.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True,  
                        arch_hpm_recurrent=True,
                        absorption_timescale=1.0/10.0,
                        diffusion_timescales=[1.05, 10., 100., 1000.],
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
if 0:
    import hpm_0102
    print "RUNNING LSTM"
    tr, va, te = hpm_0102.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
if 0:
    import gru_5_0 as gru
    print "RUNNING GRU WITH DELTA T"
    tr, va, te = gru.train_model(encoder='gru',
                        arch_gru_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
    print "RUNNING GRU WITHOUT DELTA T"
    tr, va, te = gru.train_model(encoder='gru',
                        arch_gru_include_delta_t=False,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_synthetic_cluster.npz',
                        arch_output_fn='logistic')

if 0:
    import gru_4_0 as gru
    print "RUNNING contin time GRU"
    tr, va, te = gru.train_model(encoder='ctgru',
                        arch_ctgru_rate_function='softplus',
                        arch_ctgru_init_recurrent_timescale=100.0,
                        arch_ctgru_init_forget_timescale=100.0,
                        arch_ctgru_include_delta_t=False,
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_synthetic_cluster.npz',
                        arch_output_fn='logistic')

# LSTM with retained residuals 
if 0:
    import hpm_0102
    print "RUNNING LSTM"
    tr, va, te = hpm_0102.train_model(encoder='lstm',
                        arch_lstm_include_delta_t=True,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/lstm_synthetic_cluster.npz',
                        arch_output_fn='logistic')
# SPM 4.0
if 0: 
    import gru_4_0 as spm
    print "RUNNING SPM_4_0"
    tr, va, te = spm.train_model(encoder='spm',
                        show_weights=False,
                        arch_input_map_constraint='none',
                        #arch_spm_include_input_gate=True,     v. 2.*
                        #arch_spm_include_forgetting=ifv,      v. 2.*
                        #arch_spm_input_includes_gamma=False,  v. 2.3
                        arch_spm_rate_function='softplus', # exp or softplus
                        arch_spm_init_input_timescale=.05,  
                        arch_spm_init_forget_timescale=10.,  
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20,
                        n_hid=20, 
                        data_file=filename,
                        saveto='weights/spm_synthetic_cluster.npz',
                        arch_output_fn='logistic') 
if 0: 
    import hpm_0102 as hpm
    print "RUNNING HPM 0102"
    tr,va,te = hpm.train_model(encoder='hpm',
                        show_weights=False,
                        arch_hpm_gated=True, 
                        arch_hpm_recurrent=True,
                        arch_hpm_prior_exp=False,
                        #arch_hpm_gamma_scaled_mu=True, 
                        timescales=2.0**numpy.arange(0,8),
                        valid_portion=.15,
                        valid_freq=1,
                        maxlen=1000,
                        patience=20, 
                        n_hid=20,
                        data_file=filename,
                        saveto='weights/hpm_synthetic_cluster.npz',
                        arch_output_fn='logistic')

if 0: 
    import gru_5_2 as gru
    for v in ['abcde']: #['g','h','i','j']: # ['a','b','c','d','e','f']:
        for h in [20, 40]:
            print "RUNNING CTGRU 5.2 WITH " , h , " HID and CLUSTER 6" + v
            filename = '../data/synthetic_cluster/cluster3_window6' + v
            tr,va,te = gru.train_model(encoder='ctgru',
                            show_weights=False,
                            arch_ctgru_ohat_for_sscale=False, # DEBUG
                            arch_ctgru_include_priors=False, # DEBUG
                            timescales=10**numpy.arange(-1.0,3.0,.5),
                            valid_portion=.15,
                            valid_freq=1,
                            maxlen=1000,
                            patience=20, 
                            n_hid=h,
                            data_file=filename,
                            saveto='weights/hpm_synthetic_cluster.npz',
                            arch_output_fn='logistic')
if 1:
    import gru_5_2 as gru
    for v in ['abcde']: #['g','h','i','j']: # ['a','b','c','d','e','f']:
        for h in [20, 40]:
            print "RUNNING GRU WITH " , h , " HID and WITH DELTA T, CLUSTER 6" + v
            filename = '../data/synthetic_cluster/cluster3_window6' + v
            tr,va,te = gru.train_model(encoder='gru',
                            show_weights=False,
                            arch_ctgru_ohat_for_sscale=False,
                            arch_ctgru_include_priors=False,
                            arch_gru_include_delta_t = True,
                            #timescales=10**numpy.arange(-1.0,3.0,.5),
                            valid_portion=.15,
                            valid_freq=1,
                            maxlen=1000,
                            patience=20, 
                            n_hid=h, 
                            data_file=filename,
                            saveto='weights/hpm_synthetic_cluster.npz',
                            arch_output_fn='logistic')

