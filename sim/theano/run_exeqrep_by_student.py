import numpy
n_datasets = 8
te = numpy.zeros(n_datasets)
tll = numpy.zeros(n_datasets)
tauc = numpy.zeros(n_datasets)
if 0:
    import gru_5_3 as gru
    print "RUNNING CTGRU 5.3"
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = gru.train_model(encoder='ctgru',
                            show_weights=False,
                            arch_ctgru_include_priors=False, 
                            arch_ctgru_include_smoothc=False,
                            arch_ctgru_include_delta_t=False,
                            #arch_ctgru_include_nodecay_out=True, # DEBUG 5.3_test
                            timescales=[0.0006944,0.0100117,0.1443376,2.0808957,30.0000000],
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/gru53_exeqrep.npz',
                            arch_output_fn='logistic')
    print('TEST ACC mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
    print('TEST LL  mean',tll.mean(),'SEM',tll.std()/numpy.sqrt(n_datasets))
    print('TEST AUC mean',tauc.mean(),'SEM',tauc.std()/numpy.sqrt(n_datasets))

if 1:
    import gru_5_2 as gru
    print "RUNNING GRU 5.2"
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = gru.train_model(encoder='ctgru',
                            show_weights=False,
                            arch_gru_include_delta_t=False, ##################
                            arch_ctgru_ohat_for_sscale=False, ################
                            timescales=10.0**numpy.arange(-3.0,2.5,.5),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/lstm_exeqrep.npz',
                            arch_output_fn='logistic')
if 0:
    import hpm_041317 as hpm
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = hpm.train_model(encoder='hpm',
                            show_weights=False,
                            arch_hpm_gated=True,
                            arch_hpm_recurrent=True,
                            timescales=10.0**numpy.arange(-3,3),
                            #absorption_timescale = 1./24./60./3.,
                            #diffusion_timescales=10.0**numpy.arange(1,7),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/lstm_exeqrep.npz',
                            arch_output_fn='logistic')

elif (0):
    # RUN LSTM WITH ADDITIONAL INPUTS
    if 0:
        import hpm_0102 as hpm
        for i in range(1,n_datasets+1):
            filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
            te[i-1], tll[i-1], tauc[i-1] = hpm.train_model(encoder='lstm',
                                arch_lstm_include_delta_t=True, 
                                arch_lstm_include_input_gate=True,
                                arch_lstm_include_forget_gate=True,
                                arch_lstm_include_output_gate=True,
                                valid_portion=3.0/(32.0-4.0),
                                valid_freq=5,
                                maxlen=50000,
                                patience=150,
                                n_hid=60,
                                data_file=filename,
                                saveto='weights/lstm_exeqrep.npz',
                                arch_output_fn='logistic')
    # RUN HPM
    else:
        import hpm_0102
        for i in range(1,n_datasets+1):
            filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
            te[i-1], tll[i-1], tauc[i-1]  = hpm_0102.train_model(encoder='hpm',
                                show_weights=False,
                                arch_hpm_gated=True, 
                                arch_remap_input=True,
                                arch_hpm_recurrent=True,
                                arch_hpm_prior_exp=False,
                                #arch_hpm_gamma_scaled_mu=True, 
                                timescales=2.0**numpy.arange(-11,7),
                                valid_portion=3.0/(32.0-4.0),
                                valid_freq=5,
                                maxlen=50000,
                                patience=150, 
                                n_hid=60,
                                saveto='weights/hpm_0102_exeqrep.npz',
                                data_file=filename,
                                arch_output_fn='logistic')
elif (0):
    import hpm_030517
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1]  = hpm_030517.train_model(encoder='hpm',
                            show_weights=False,
                            arch_hpm_gated=True, 
                            arch_input_map_constraint='none',
                            arch_hpm_alpha_constraint='none', # DEBUG
                            arch_hpm_recurrent=True,
                            arch_hpm_prior_exp=False,
                            timescales=2.0**numpy.arange(-11,7),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150, 
                            n_hid=60,
                            saveto='weights/hpm_030517_exeqrep.npz',
                            data_file=filename,
                            arch_output_fn='logistic')

elif (0):
    import hpm_031117

    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1]  = hpm_031117.train_model(encoder='hpm',
                            show_weights=False,
                            arch_hpm_gated=True, 
                            arch_input_map_constraint='none',
                            arch_hpm_alpha_exp=0.0,
                            arch_hpm_mu_exp=0.1,
                            arch_hpm_prior_exp=0.9,
                            arch_hpm_recurrent=True,
                            timescales=2.0**numpy.arange(-11,7),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150, 
                            n_hid=240, # DEBUG *********************
                            saveto='weights/hpm_031117_exeqrep.npz',
                            data_file=filename,
                            arch_output_fn='logistic')
elif (0): # 1-1 connections
    import hpm_031117

    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1]  = hpm_031117.train_model(encoder='hpm',
                            show_weights=False,
                            arch_hpm_gated=True, 
                            arch_input_map_constraint=.5,
                            arch_hpm_alpha_exp=0.0,
                            arch_hpm_mu_exp=0.1,
                            arch_hpm_prior_exp=0.9,
                            arch_hpm_recurrent=False, # ?????
                            timescales=2.0**numpy.arange(-11,7),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150, 
                            n_hid=60,
                            saveto='weights/hpm_031117_exeqrep.npz',
                            data_file=filename,
                            arch_output_fn='logistic')
elif (0):
    import hpm_0102
    # LSTM WITH NO INPUT, FORGET, OUTPUT GATE
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = hpm_0102.train_model(encoder='lstm',
                            arch_lstm_include_delta_t=True, 
                            arch_lstm_include_input_gate=False,
                            arch_lstm_include_forget_gate=False,
                            arch_lstm_include_output_gate=False,
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/lstm_exeqrep.npz',
                            arch_output_fn='logistic')

if (0): # rob's survival process memory
    import spm 
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_student_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = spm.train_model(encoder='spm',
                            arch_spm_init_forget_timescale=100.0,
                            arch_spm_init_input_timescale=1.0/24.0/5.0,
                            arch_spm_include_input_gate=True,
                            arch_spm_rate_function='exp',
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/spm_exeqrep.npz',
                            arch_output_fn='logistic')
print(1-te)
print(tll)
print(tauc)

print('TEST ACC mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
print('TEST LL  mean',tll.mean(),'SEM',tll.std()/numpy.sqrt(n_datasets))
print('TEST AUC mean',tauc.mean(),'SEM',tauc.std()/numpy.sqrt(n_datasets))


