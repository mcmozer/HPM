import numpy
import hpm_0416A

# RUN LSTM WITH ADDITIONAL INPUTS
if 0:
    n_datasets = 8
    tr = numpy.zeros(n_datasets)
    te = numpy.zeros(n_datasets)
    va = numpy.zeros(n_datasets)
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_unlabeled_studentitem_' + str(i)
        tr[i-1], va[i-1], te[i-1] = hpm_0416A.train_model(encoder='lstm',
                            arch_lstm_include_delta_t=True,
                            valid_freq=1,
                            patience=25,
                            n_hid=20,
                            data_file=filename,
                            arch_output_fn='logistic')
        print(1-te)
    print('TEST ERROR mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))

    n_datasets = 8
    tr = numpy.zeros(n_datasets)
    te = numpy.zeros(n_datasets)
    va = numpy.zeros(n_datasets)
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_unlabeled_studentitem_' + str(i)
        tr[i-1], va[i-1], te[i-1] = hpm_0416A.train_model(encoder='lstm',
                            arch_lstm_include_delta_t=False, 
                            valid_freq=1,
                            patience=25,
                            n_hid=20,
                            data_file=filename,
                            arch_output_fn='logistic')
        print(1-te)
    print('TEST ERROR mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))

# RUN HPM
if 1:
    n_datasets = 8
    tr = numpy.zeros(n_datasets)
    te = numpy.zeros(n_datasets)
    va = numpy.zeros(n_datasets)
    for i in range(1,n_datasets+1):
        filename = '../data/exeqrep/exeqrep_per_unlabeled_studentitem_' + str(i)
        tr[i-1], va[i-1], te[i-1] = hpm_0416A.train_model(encoder='hpm',
                            arch_remap_input=True,
                            arch_hpm_recurrent=True,
                            timescales=2.0**numpy.arange(-11,7),
                            valid_freq=5,
                            patience=25,
                            n_hid=20, 
                            data_file=filename,
                            arch_output_fn='logistic')
        print(1-te)
    print('TEST ERROR mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))

