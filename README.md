# Complex Convolutions
## About
This repository contains code to reproduce results for the submission titled, "Complex-Valued Convolutions for Modulation Recognition using Deep Learning" which was accepted into the 2020 IEEE ICC: Open Workshop On Machine Learning In Communications. 

## Code*
### Scripts
There is a folder title 'Complex_Convolutions', which contains scripts to execute and recreate the results in the paper.

The following code will execute the experiment: (be sure to include the path to the dataset)
```
python3 run.py --dataset RML2016 --data_directory <path_to_data> --train_pct 50 --train_SNRs -20 20 2 --test_SNRs -20 20 2 --load_weights False 
```

### Jupyter Notebook
There is a Jupyter Notebook that trains the networks described in the paper as well as recreates all the plots used in the Results section.

*All code is written in Keras

## Data
Data for this submission (RML2016.10b.tar.bz2) can be found at: https://www.deepsig.io/datasets. To ensure proper execution of the code, be sure the data is saved as 'RML2016.10a_dict.pkl'.
