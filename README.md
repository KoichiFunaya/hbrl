# How to use the HBRL experiment scripts

18.01.2021
Koichi Funaya


### Install libraries

run the following scripts from the command line, to install necessary python library requirements.

> pip install -r requirements.txt<br>

### Place the raw EMBER data in the data folder.

Obtain data from the following URL.

> https://pubdata.endgame.com/ember/ember_dataset_2018_2.tar.bz2<br>

Decompress them and store in a directory of your choice.

### Modify the configuration files.

There are toml files in the following directory.

> ./config/ijcai_2021/.

For example, use an editor, modify the config file below.

> ./config/ijcai_2021/prepare_data.toml

Modify the following directory paths.

> DataPath.data_dir  --> destination folder
> EMBER_LGBM.ember_dir --> source folder

### Run Step 1

Run the following Jupyter notebook, to generate cross-validation data.

> ./exp_ijcai/step1_load-data_train-lgbm.ipynb

Note the raw file name that the scripts have stored the five-fold train-validation data.

Create another config file, e.g. by modifying the file below.

> ./config/ijcai_2021/test_HBRL.toml

The raw file name should be written in the field below.

> DataPath.raw_file_name 

### Run Step 2

Run the following Jupyter notebook, to generate a dataset for HBRL.

> ./exp_ijcai/step2_convert-dataset.ipynb

Note the data file name and write in the same toml file as in Step 1, at the field below.

> ./config/ijcai_2021/test_HBRL.toml

> DataPath.data_file_name 


### Run Step 3

Run the following Jupyter notebook. You get a HBRL results.

> ./exp_ijcai/step3_train-test_HBRL.ipynb

To ensure the repeatability of the experiment, we set the random seed in the config file.

> ./config/ijcai_2021/test_HBRL.toml

> HyperParameters.random_seed 


### Run Step 4

Run the following Jupyter notebook to see the result from the HBRL results.

> ./exp_ijcai/step4_visualize-results_HBRL.ipynb

## Run Step 5

Run the following Jupyter notebook to see the ROC AUC area plot.

>  ./exp_ijcai/step5_plot-AUC.ipynb

