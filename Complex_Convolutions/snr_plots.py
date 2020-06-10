def plot(test_snrs, snr_accs, model_names, tag):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    Visualize the performance of the various models as a function of SNR.
    Quantify and visualize the performance benfeits of one model over another.
    
    # Inputs
    test_snrs   - (int) List of SNRs to be tested
    snr_accs    - Array of classification accuracies at each SNR, for each architecture
    model_names - (str) List of built architecture names
    tag         - (str) Namimg convention for the experiment
    
    # Outputs
    Two saved scatter plots: 
    (1) accuracy @ every SNR over all models
    (2) % differnce in accuracy @ every SNR versus a model of interest
    """
    # Import Packages
    import os
    import matplotlib.pyplot as plt

    # Plot the performance over all modulations as a funciton of SNR
    plt.figure()
    for i in range(len(model_names)):
        plt.scatter(range(test_snrs[0],test_snrs[-1]+2,2), snr_accs[i]*100, label = model_names[i])
    plt.grid()
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.getcwd()+'/'+tag+'/Figures/SNR_Classification_Accuracy.png',transparent = True, bbox_inches = 'tight', pad_inches = 0.01)
    
    # Plot the % difference in performace with respect to the last model
    plt.figure()
    for i in range(len(model_names)-1):
        plt.scatter(range(test_snrs[0],test_snrs[-1]+2,2), (snr_accs[-1]-snr_accs[i])/snr_accs[i]*100, label = model_names[i])
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Improvement (%)')
    plt.grid()
    plt.legend()
    plt.savefig(os.getcwd()+'/'+tag+'/Figures/SNR_Classification_Improvement_' + model_names[-1] + '.png',transparent = True, bbox_inches = 'tight', pad_inches = 0.01)