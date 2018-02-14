import numpy
import hpm_0416A

# LSTM
if 1:
    n_datasets = 10
    tr = numpy.zeros(n_datasets)
    te = numpy.zeros(n_datasets)
    va = numpy.zeros(n_datasets)
    for i in range(1,n_datasets+1):
        filename = '../data/type_token/hp_4streams_5tokens_' + str(i)
        tr[i-1], va[i-1], te[i-1] = hpm_0416A.train_model(encoder='lstm',
                            type_token_sim=True,
                            arch_lstm_include_delta_t=False, # DEBUG****
                            n_hid=25, 
                            data_file=filename)
        print(1-te)
    print('TEST ERROR mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))

# HPM
if 0:
    n_datasets = 10
    tr = numpy.zeros(n_datasets)
    te = numpy.zeros(n_datasets)
    va = numpy.zeros(n_datasets)
    for i in range(1,n_datasets+1):
        filename = '../data/type_token/hp_4streams_5tokens_' + str(i)
        tr[i-1], va[i-1], te[i-1] = hpm_0416A.train_model(encoder='hpm',
                            type_token_sim=True,
                            timescales=[1.,2.,4.,8.,16.,32.,64.],
                            n_hid=25, 
                            data_file=filename)
        print(1-te)
    print('TEST ERROR mean',(1-te).mean(),'SEM',(1-te).std()/numpy.sqrt(n_datasets))
