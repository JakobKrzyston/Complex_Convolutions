def compute(model_names, Models, classes, tag, mods):
    """
    # Author
    Jakob Krzyston
    
    # Purpose
    Compute and save the inputs that result in one-hot classificaiton for a given architecture and modulation
    Plot these 'max-inputs', aka activation maximizations, and save the plots
    
    # Inputs
    model_names - (str) List of built architecture names
    Models      - Built architectures
    classes     - (str) List on modulation classes
    tag         - (str) Namimg convention for the experiment
    mods        - (str) List of the modulation names
    
    # Outputs
    Saved activation maximizations
    Saved plot of the activation maximizations
    """
    # Import Packages
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    import vis
    from vis.visualization import visualize_activation
    from vis.utils import utils
    import matplotlib.pyplot as plt
    from keras import activations
    
    # Compute the maximizing inputs 
    max_inputs = np.zeros((len(model_names),len(classes),2,128))
    mod_idx = np.arange(len(classes))
    for i in range(len(model_names)):
        model = Models[i]
        layer_index = -3
        model.layers[-2].activation = activations.linear
        model = utils.apply_modifications(model)
        for vis_num in mod_idx:
            visualization = visualize_activation(model, layer_index, filter_indices=vis_num, input_range=(-1.0, 1.0))
            max_inputs[i,vis_num,0,:] = visualization[0]
            max_inputs[i,vis_num,1,:] = visualization[1]
    
    # Save max_inputs
    np.save(os.getcwd()+'/'+tag+'/Computed_Values/Activation_Maximizations', max_inputs.astype('float32'))
    
    # Plot the activations
    plt.figure(figsize = (28,30))
    for i in range(len(mod_idx)):
        for j in range(3):
            ax = plt.subplot2grid((len(mod_idx),4), (i,j))
            plt.plot((max_inputs[j,mod_idx[i],0,:]), label = 'I')
            plt.plot((max_inputs[j,mod_idx[i],1,:]), label = 'Q')
            ax.axes.get_xaxis().set_visible(False)
            plt.tight_layout()
            plt.legend()
            plt.title(model_names[j]+': '+mods[mod_idx[i]], fontdict={'fontsize': 20})

    # Save the figure
    plt.savefig(os.getcwd()+'/'+tag+'/Figures/Activation_Maximizations.png',transparent = True, bbox_inches = 'tight', pad_inches = 0.02)
