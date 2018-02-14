import numpy

n_datasets = 8
te = numpy.zeros(n_datasets)
tll = numpy.zeros(n_datasets)
tauc = numpy.zeros(n_datasets)

if 1:
    import gru_5_1 as gru
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = gru.train_model(encoder='ctgru',
                            arch_ctgru_ohat_for_sscale=True, ###############
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            timescales=[0.0006944,0.0100117,0.1443376,2.0808957,30.0000000],
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/ctgru_exeqrep.npz',
                            arch_output_fn='logistic')

    print('TEST ACC mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
    print('TEST LL  mean',tll.mean(),'SEM',tll.std()/numpy.sqrt(n_datasets))
    print('TEST AUC mean',tauc.mean(),'SEM',tauc.std()/numpy.sqrt(n_datasets))

    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = gru.train_model(encoder='ctgru',
                            arch_ctgru_ohat_for_sscale=False, ###############
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            timescales=[0.0006944,0.0100117,0.1443376,2.0808957,30.0000000],
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/ctgru_exeqrep.npz',
                            arch_output_fn='logistic')

    print('TEST ACC mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
    print('TEST LL  mean',tll.mean(),'SEM',tll.std()/numpy.sqrt(n_datasets))
    print('TEST AUC mean',tauc.mean(),'SEM',tauc.std()/numpy.sqrt(n_datasets))

    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = gru.train_model(encoder='gru',
                            arch_gru_include_delta_t = True, ##############
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150,
                            n_hid=60,
                            data_file=filename,
                            saveto='weights/ctgru_exeqrep.npz',
                            arch_output_fn='logistic')


# RUN LSTM WITH ADDITIONAL INPUTS
if 0:
    import hpm_0102
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
        te[i-1], tll[i-1], tauc[i-1] = hpm_0102.train_model(encoder='lstm',
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
if 0:
    import hpm_0102
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
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
if 0:
    import hpm_030517
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_studentitem_' + str(i)
        te[i-1], tll[i-1], tauc[i-1]  = hpm_030517.train_model(encoder='hpm',
                            show_weights=False,
                            arch_hpm_gated=True, 
                            arch_input_map_constraint='none',
                            arch_hpm_alpha_constraint='none',
                            arch_hpm_recurrent=True,
                            arch_hpm_prior_exp=False,
                            timescales=2.0**numpy.arange(-11,7),
                            valid_portion=3.0/(32.0-4.0),
                            valid_freq=5,
                            maxlen=50000,
                            patience=150, 
                            n_hid=60,
                            saveto='weights/hpm_0102_exeqrep.npz',
                            data_file=filename,
                            arch_output_fn='logistic')

print(1-te)
print(tll)
print(tauc)

print('TEST ACC mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
print('TEST LL  mean',tll.mean(),'SEM',tll.std()/numpy.sqrt(n_datasets))
print('TEST AUC mean',tauc.mean(),'SEM',tauc.std()/numpy.sqrt(n_datasets))
