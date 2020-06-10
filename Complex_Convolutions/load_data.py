def load(dataset,
         where_data,
         samples,
         train_pct,
         train_SNR_h,
         train_SNR_l,
         train_SNR_step,
         test_SNR_h,
         test_SNR_l,
         test_SNR_step,
         train_test_SNR_eq):
    """
    # Author
    Jakob Krzyston (jakobk@gatech.edu)

    # Purpose
    Load dataset, abiding specific parameters to enable the desired experiment to be conducted

    # Inputs
    dataset           - (str) Specify which dataset, choices include 'RML2016', 'Real'
    where_data        - (str) Path to the dataset
    samples           - (int) # of sampled to be used per modulaiton per dB, only used for 'Real' dataset
    train_pct         - (double) Percentage of data to be used for training, e.g. 0.3 is 30% training
    train_SNR_h       - (int) Upper bound dB to be used in the training set
    train_SNR_l       - (int) Lower bound dB to be used in the training set
    train_SNR_step    - (int) Step size between dB to be used int he training set
    test_SNR_h        - (int) Upper bound dB to be used in the testing set
    test_SNR_l        - (int) Lower bound dB to be used in the testing set
    test_SNR_step     - (int) Step size between dB to be used int he testingg set
    train_test_SNR_eq - (Bool) Specify whether the bounds for the testing and training data are equivalent

    # Outputs
    test_snrs    - Array of SNRs used inthe testing dataset
    mods         - List of the modulation types
    X_train1     - Data used to train the networks
    X_test1      - Data used to test the networks
    Y_train1     - Labels used to train the networks
    Y_test1      - Labels used to test the networks
    X_test2      - (if train_test_SNR_eq == False) Data used to evaluate trained networks
    Y_test2      - (if train_test_SNR_eq == False) Labels used to evaluate trained networks
    test_samples - SNR organized data for later analysis
    test_labels  - SNR organized labels for later analysis

    """
    # Import packages
    import numpy as np
    import pickle

    #define function to make
    def to_onehot(yy):
        yy1 = np.zeros([len(yy),max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1

    # Load data
    # Specify the where the datasets are
    data_directory = where_data

    if dataset == 'RML2016': #data is from https://www.deepsig.io/datasets
        print('RML2016')
        print('Train pct: ' + str(train_pct))

        Xd = pickle.load(open(data_directory+"RML2016.10a_dict.pkl", 'rb'), encoding = 'latin1')
        test_snrs,mods = map(lambda j: sorted( list( set( map( lambda x: x[j], Xd.keys() ) ) ) ), [1,0])

    if dataset == 'Real': #Wireless Link
        print('Wireless Link (Real) Data')
        print('Samples/mod/dB: ' + str(samples))
        print('Train pct: ' + str(train_pct))

        Xd = pickle.load(open(data_directory+"Wireless_Link_-20_40_2.dat", 'rb'), encoding = 'latin1')
        test_snrs,mods = map(lambda j: sorted( list( set( map( lambda x: x[j], Xd.keys() ) ) ) ), [1,0])

    # Parse data according to experimental specifactions 
    if train_test_SNR_eq == True:
        print('Train and Test SNRs are equal: ' + str(range(train_SNR_l,train_SNR_h,train_SNR_step)))

        X = []
        lbl = []

        for mod in mods:
            for snr in range(train_SNR_l,train_SNR_h,train_SNR_step):
                X.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
        X = np.vstack(X)

        # Partition Data
        np.random.seed(2020)
        n_examples = X.shape[0]
        n_train    = int(round(n_examples * train_pct))
        train_idx  = np.random.choice(range(0,n_examples), size=n_train, replace=False)
        test_idx   = list(set(range(0,n_examples))-set(train_idx))
        X_train1   = X[train_idx]
        X_test1    = X[test_idx]
        print('Train set size: ' + str(X_train1.shape[0]))
        print('Test set size: ' + str(X_test1.shape[0]))
        Y_train1 = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        Y_test1  = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

        # Pull out the data by SNR
        labels_oh = np.eye(len(mods))
        test_samples = np.zeros((len(test_snrs),len(mods)*samples, 2, 128))
        test_labels  = np.zeros((len(test_snrs),len(mods)*samples, len(mods)))
        for i in range(len(test_snrs)):
            for j in range(len(mods)):
                test_samples[i,j*samples:(j+1)*samples,:,:] = Xd[(mods[j],test_snrs[i])]
                test_labels[i,j*samples:(j+1)*samples,:]    = labels_oh[j]

        in_shp = list(X_train1.shape[1:])
        classes = mods

        # Account for the train and test SNRs being the same
        X_test2 = None
        Y_test2 = None

    else:
        print('Train SNRs: ' + str(range(train_SNR_l,train_SNR_h,train_SNR_step)))
        print('Test SNRs: ' + str(range(test_SNR_l,test_SNR_h,test_SNR_step)))

        X1 = []
        lbl1 = []

        X2 = []
        lbl2 = []

        for mod in mods:
            for snr in range(train_SNR_l,train_SNR_h,train_SNR_step):
                X1.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  lbl1.append((mod,snr))
        X1 = np.vstack(X1)

        # Partition Data
        np.random.seed(2020)
        n_examples = X1.shape[0]
        n_train    = int(round(n_examples * train_pct))
        train_idx1 = np.random.choice(range(0,n_examples), size=n_train, replace=False)
        test_idx1  = list(set(range(0,n_examples))-set(train_idx1))
        X_train1   = X1[train_idx1]
        X_test1    = X1[test_idx1]

        print('Train set size: ' + str(X_train1.shape[0]))
        print('Train-eval set size: ' + str(X_test1.shape[0]))

        Y_train1 = to_onehot(list(map(lambda x: mods.index(lbl1[x][0]), train_idx1)))
        Y_test1  = to_onehot(list(map(lambda x: mods.index(lbl1[x][0]), test_idx1)))

        in_shp = list(X_train1.shape[1:])
        classes = mods

        for mod in mods:
            for snr in range(test_SNR_l,test_SNR_h,test_SNR_step):
                X2.append(Xd[(mod,snr)])
                for i in range(Xd[(mod,snr)].shape[0]):  lbl2.append((mod,snr))
        X2 = np.vstack(X2)

        test_idx2 = np.random.choice(range(0,X2.shape[0]), size=X2.shape[0], replace=False)
        X_test2   = X2[test_idx2]
        Y_test2   = to_onehot(list(map(lambda x: mods.index(lbl2[x][0]), test_idx2)))
        print('Test set size: ' + str(X_test2.shape[0]))
        
        # Pull out the data by SNR
        labels_oh = np.eye(len(mods))
        test_samples = np.zeros((len(test_snrs),len(mods)*samples, 2, 128))
        test_labels  = np.zeros((len(test_snrs),len(mods)*samples, len(mods)))
        for i in range(len(test_snrs)):
            for j in range(len(mods)):
                test_samples[i,j*samples:(j+1)*samples,:,:] = Xd[(mods[j],test_snrs[i])]
                test_labels[i,j*samples:(j+1)*samples,:]    = labels_oh[j]

        test_snrs = np.arange(test_SNR_l,test_SNR_h,test_SNR_step)
    
    return test_snrs, mods, X_train1, X_test1, Y_train1, Y_test1, X_test2, Y_test2, test_samples, test_labels, classes, in_shp 