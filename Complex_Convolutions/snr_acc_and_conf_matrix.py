def eval(test_snrs, model, model_name, test_samples, test_labels, tag, classes, batch_size):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    For a given architecture, @ each SNR in the test set: 
    (1) Compute the accuracy 
    (2) Compute the confusion matrix  
    
    # Inputs
    test_snrs    - (int) List of SNRs to be tested
    model        - Built architecture 
    model_name   - (str) Name of the built architecture
    test_samples - Testing data organized by SNR
    test_labels  - Testing labels organized by SNR
    tag          - (str) Namimg convention for the experiment
    classes      - (str) List on modulation classes
    batch_size   - (int) batch_size
    
    # Outputs
    Saved accuracies and confusion matrices at every SNR
    """
    # Import Packages
    import os
    import numpy as np
    
    # Function to plot the confusion matrices
    import confusion_matrix

    # Compute Acuracy and plot confusion matrices by SNR
    acc = np.zeros(len(test_snrs))
    for s in range(len(test_snrs)):

        # extract classes @ SNR
        test_X_i = test_samples[s]
        test_Y_i = test_labels[s]
        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        # Confusion matrix @ all SNRs in the test set
        if s == 0 or s == len(test_snrs)-1:
            filename = os.getcwd()+'/'+tag+'/Figures/'+ model_name + '_SNR_' + str(test_snrs[s]) + '_Confusion_Matrix.png'
            confusion_matrix.plot(confnorm, tag, model_name, filename, labels=classes)
        
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        acc[s] = 1.0*cor/(cor+ncor)
    # Save results
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/' + model_name + '_SNR_Accuracy', acc.astype('float32'))
    print(model_name + ' Accuracy:\n' + str(acc))