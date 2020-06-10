def eval(model, model_name, X_test, Y_test, tag, classes, batch_size):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    Compute the overall accuracy and confusion materix for a given architecture. 
    Overall means averaged across SNRs in the specified test set.
    
    # Inputs
    model      - Built architecture 
    model_name - (str) Name of the built architecture
    X_test     - Testing data
    Y_test     - Testing labels 
    tag        - (str) Namimg convention for the experiment
    classes    - (str) List on modulation classes
    batch_size - (int) batch_size
    
    # Outputs
    Saved overall accuracy and overall confusion matrix
    """
    # Import Packages
    import os
    import numpy as np
        
    # Function to plot the confusion matrices
    import confusion_matrix
    
    # Compute the confusion matrices
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,X_test.shape[0]):
        j = list(Y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    cor  = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc  = 1.0*cor/(cor+ncor)
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/' + model_name + '_Overall_Accuracy', acc.astype('float32'))
    print("Overall Accuracy - " + model_name + ": ", acc)
    filename = os.getcwd()+'/'+tag+'/Figures/'+ model_name + '_Overall_Confusion_Matrix.png'
    confusion_matrix.plot(confnorm, tag, model_name, filename, labels=classes)