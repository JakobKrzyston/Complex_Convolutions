"""
# Author
Jakob Krzyston (jakobk@gatech.edu)

# Last Updated
June 10, 2020

# Purpose
Enable rapid, organized experimentation of algorithms for I/Q data processing

"""
## Import packages and functions
import os, argparse
import numpy as np
import load_data, build_models, train_models
import overall_acc_and_conf_matrix, snr_acc_and_conf_matrix
import snr_plots, activation_maximizations


## Handle input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--data_directory', type=str, required=True)
parser.add_argument('--samples', type=int, required=False, default=1000)
parser.add_argument('--train_pct', type=int, required=True)
parser.add_argument('--train_SNRs', type=int, nargs='+', required=True)
parser.add_argument('--test_SNRs', type=int, nargs='+', required=True)
parser.add_argument('--load_weights', type=str, required=True)
args = parser.parse_args()

## Experimental parameters
# Extract the name of the dataset
dataset = args.dataset

# Determine the train/test split
train_pct = (100-args.train_pct)/100

# Number of samples from each class per dB SNR to be used
samples = args.samples

# Specify SNRs to conduct the experiment
train_SNR_l = args.train_SNRs[0]
train_SNR_h = args.train_SNRs[1]
train_SNR_step = args.train_SNRs[2]

test_SNR_l = args.test_SNRs[0]
test_SNR_h = args.test_SNRs[1]
test_SNR_step = args.test_SNRs[2]

# Determine
if train_SNR_l == test_SNR_l and train_SNR_h == test_SNR_h and train_SNR_step == test_SNR_step:
    train_test_SNR_eq = True
else:
    train_test_SNR_eq = False

# If already trained, save time and load the saved weights
load_weights = args.load_weights

# Specify file tag to ID the results from this run
# If splitting the data by SNR, include the SNR bounds that were used to train the network in the tag. 
# The bounds should not overlap as this would defeat the purpose of the experiment
tag = dataset+'_train_'+str(train_SNR_l)+'_'+str(train_SNR_h)+'_test_'+str(test_SNR_l)+'_'+str(test_SNR_h)

# Setup directories to organize results 
sub_folders = ['Weights', 'Figures', 'Computed_Values']
for i in range(len(sub_folders)):
    path = os.path.join(os.getcwd(),tag+'/'+sub_folders[i])
    os.makedirs(path)


## Load data
test_snrs,mods,X_train1,X_test1,Y_train1,Y_test1,X_test2,Y_test2,test_samples,test_labels,classes,in_shp = load_data.load(dataset,args.data_directory,samples,train_pct,train_SNR_h,train_SNR_l,train_SNR_step,test_SNR_h,test_SNR_l,test_SNR_step,train_test_SNR_eq) 


## Build the architectures
Models, model_names = build_models.build(in_shp, classes, dr = 0.5)


## Train/ load weights for the built models
# If training, will plot and save the training & validation loss curves in the 'Figures' folder
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    train_models.train(model, model_name, X_train1, X_test1, Y_train1, Y_test1, tag, load_weights, epochs=100, batch_size = 1024)
    
    
## Verify which dataset to use when testing from here on, should it belong to a different distribution than the training data
if train_test_SNR_eq == True:
	X_test = X_test1
	Y_test = Y_test1
else:
	X_test = X_test2
	Y_test = Y_test2
    
    
## Overall accuracy and confusion matrix for the corresponding models
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    overall_acc_and_conf_matrix.eval(model, model_name, X_test, Y_test, tag, classes, batch_size = 1024)
    

## Accuracy and confusion matrix for the corresponding models at each SNR
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    snr_acc_and_conf_matrix.eval(test_snrs, model, model_name, test_samples, test_labels, tag, classes, batch_size=1024)
    

## Compoute the classification accuracy by SNR for each model
# Extract the accuracy by SNR for each model
snr_accs = np.zeros((len(model_names), len(test_snrs)))
for i in range(len(model_names)):
    model_name = model_names[i]    
    acc_name = os.getcwd() + '/' + tag + '/Computed_Values/' + model_name + '_SNR_Accuracy.npy'
    # Extra elements are added to the array in the saving process (not sure why)
    # This will remove those elements and include the correct values 
    snr_accs[i,:] = np.fromfile(acc_name, dtype = 'float32')[-int(len(test_snrs)):]
# Compute SNR accuracies
snr_plots.plot(test_snrs, snr_accs, model_names, tag)


# Compute and save the activation maximizations
activation_maximizations.compute(model_names, Models, classes, tag)