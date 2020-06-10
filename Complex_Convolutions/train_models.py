def train(model, model_name, X_train1, X_test1, Y_train1, Y_test1, tag, load_weights, epochs=100, batch_size = 1024):
    """
    # Author
    Jakob Krzyston

    # Purpose
    Train a network and save the weights to the created 'Weights' folder within this experiment. 

    # Inputs
    model        - Built architecture 
    model_name   - (str) Name of the built architecture
    X_train1     - Training dataset
    X_test1      - Validation/testing data
    Y_train1     - Training labels
    Y_test1      - Validation/testing labels
    epochs       - (int) Maximum number of eposh to optimize over
    batch_size   - (int) Batch size
    tag          - (str) Namimg convention for the experiment
    load_weights - (bool) Whether previously trained wieghts will be used or not
    
    # Outputs
    Saved weights to the corresponding 'Weights' folder
    """

    # Import packages
    import os, time
    import keras
    import matplotlib.pyplot as plt
    
    print('Training: '+ model_name)
    # Train the networks
    start = time.time()
    filepath = os.getcwd()+'/'+tag+'/Weights/'+model_name+'.wts.h5'
    if load_weights == True:
        model.load_weights(filepath)
    else:
        history = model.fit(X_train1,
                            Y_train1,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(X_test1, Y_test1),
                            class_weight='auto',
                            callbacks = [keras.callbacks.ModelCheckpoint(filepath,
                                                                         monitor='val_loss',
                                                                         verbose=0,
                                                                         save_best_only=True,
                                                                         save_weights_only=True,
                                                                         mode='auto'),
                                         keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       patience=5, 
                                                                       verbose=0, 
                                                                       mode='auto')
                            ])
        
        model.load_weights(filepath)
        model.save(filepath)
    end = time.time()
    duration = end - start
    print(model_name + ' Training time = ' + str(round(duration/60,5)) + 'minutes')
    
    # Plot the train/valid loss curves, save plot
    plt.figure(figsize = (7,7))
    plt.plot(history.epoch, history.history['loss'], label=model_name+": Training Error + Loss")
    plt.plot(history.epoch, history.history['val_loss'], label=model_name+": Validation Error")
    plt.xlabel('Epoch')
    plt.ylabel('% Error')
    plt.legend()
    plt.savefig(os.getcwd()+'/'+tag+'/Figures/'+model_name+'_train_validation_losses.png',transparent = True, bbox_inches = 'tight', pad_inches = 0.01)