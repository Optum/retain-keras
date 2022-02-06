# RETAIN-Keras: Keras reimplementation of RETAIN

[RETAIN is a neural network architecture originally introduced by Edward Choi](https://arxiv.org/abs/1608.05745) that enables the creations of highly interpretable Recurrent Neural Network models for patient diagnosis without any loss in model performance.  This repository holds the [keras](https://www.tensorflow.org/api_docs/python/tf/keras) reimplementation of RETAIN (originally in Theano) that allows for flexible modifications to the original code, introduces multiple new features, and increases the speed of training. RETAIN has shown to be highly effective for creating predictive models for a multitude of conditions and we are excited to share this implementation to the broader healthcare data science community.

### Improvements and Extra Features

- Simple Keras code with Tensorflow backend
- Ability to use extra numeric inputs of fixed size that can hold numeric information about a patient's visit such as age, quantity of drug prescribed, or blood pressure
- Improved embedding logic that avoids using large dense inputs
- Ability to evaluate models during training
- Ability to train models with only positive embedding contributions which improves performance
- Extra script to evaluate the model and output several helper graphics

### Installing RETAIN-Keras and Building the Environment

To run the scripts in this repository, create a Python 3.7.9 virtual environment and install the dependencies in `requirements.txt`.  We recommend using [Anaconda](https://www.anaconda.com/products/individual) to create your environment with the following commands:

```
git clone https://github.com/Optum/retain-keras.git
conda create --name=retain python=3.7.9
conda activate retain
pip install -r requirements.txt
```

### Running the code

- **Training**: - `python3 retain_train.py --num_codes=x --additional arguments`
- **Evaluating**: - `python3 retain_evaluation.py --additional arguments`
- **Interpretation**: - `python3 retain_interpretations.py`

### Training Arguments

The `retain_train.py` script will train the RETAIN model and evaluate/save it after each epoch.  The script has multiple arguments to customize the training and model:

- `--num_codes`: Integer Number of medical codes in the data set (required)
- `--numeric_size`: Integer Size of numeric inputs, 0 if none. Default: 0
- `--use_time`: Enables the extra time input for each visit. Default: off
- `--emb_size`: Integer Size of the embedding layer. Default: 200
- `--epochs`: Integer Number of epochs for training. Default: 1
- `--n_steps`: Integer Maximum number of visits after which the data is truncated. This features helps to conserve GPU Ram (only the most recent n_steps will be used). Default: 300
- `--recurrent_size`': Integer Size of the recurrent layers. Default: 200
- `--path_data_train`: String Path to train data. Default: 'data/data_train.pkl'
- `--path_data_test`: String Path to test/validation data. Default: 'data/data_test.pkl'
- `--path_target_train`: String Path to train target. Default: 'data/target_train.pkl'
- `--path_target_test`: String Path to test/validation target. Default: 'data/target_test.pkl'
- `--batch_size`: Integer Batch Size for training. Default: 32
- `--dropout_input`: Float Dropout rate for embedding of codes and numeric features (0 to 1). Default: 0.0
- `--dropout_context`: Float Dropout rate for context vector (0 to 1). Default: 0.0
- `--l2`: Float L2 regularization value for layers. Default: 0.0
- `--directory`: String Directory to save the model and the log file to. Default: 'Model' (directory needs to exist otherwise error will be thrown)
- `--allow_negative`: Allows negative weights for embeddings/attentions (original RETAIN implementation allows it but forcing non-negative weights have shown to perform better on a range of tasks). Default: off

### Evaluation Arguments

The `retain_evaluation.py` script will evaluate the specific RETAIN model and create some sample graphs.  Arguments include:

- `--path_model`: Path to the model to evaluate. Default: 'Model/weights.01.hdf5'
- `--path_data`: Path to evaluation data. Default: 'data/data_test.pkl'
- `--path_target`: Path to evaluation target. Default: 'data/target_test.pkl'
- `--omit_graphs`: Does not output graphs if argument is present. Default: (Graphs are output)
- `--n_steps`: Integer Maximum number of visits after which the data is truncated. This features helps to conserve GPU Ram (only the most recent n_steps will be used). Default: 300
- `--batch_size`: Batch size for prediction (higher values are generally faster). Default: 32

### Interpretation Arguments

The `retain_interpretations.py` script will compute probabilities for all patients and then will allow the user to select patients by ID to see specific risk scores and interpret visits (displayed as pandas dataframes).  It is highly recommended to extract this script to a notebook to enable more dynamic interaction. Arguments include:

- `--path_model`: Path to the model to evaluate. Default: 'Model/weights.01.hdf5'
- `--path_data`: Path to evaluation data. Default: 'data/data_test.pkl'
- `--path_dictionary`: Path to dictionary that maps code index to the specific alphanumeric value. If numerics inputs are used they should have indexes num_codes+1 through num_codes+numeric_size, num_codes index is reserved for padding.
- `--batch_size`: Batch size for prediction (higher values are generally faster). Default: 32

### Data and Target Format

By default the data has to be saved as a pickled pandas dataframe with the following format:

- Each row is 1 patient.
- Rows are sorted by the number of visits a person has. People with the least visits should be in the beginning of the dataframe and people with the most visits at the end.
- Column 'codes' is a list of lists where each sublist are codes for the individual visit. Lists have to be ordered by their order of events (from old to new).
- Column 'numerics' is a list of lists where each sublist contains numeric values for an individual visit. Lists have to be ordered by their order of events (from old to new). Lists have to have a static size of `numeric_size` indicating number of different numeric features for each visit. Numeric information can include things like patients age, blood pressure, BMI, length of the visit, or cost charged (or all at the same time!). This column is not used if `numeric_size` is 0.
- Column 'to_event' is a list of values indicating when the respective visit happened. Values have to be ordered from oldest to newest. This column is not used if `use_time` is not specified.

By default the target has to be saved as a pickled pandas dataframe with the following format:

- Each row is 1 patient corresponding to the patient from data file
- Column 'target' is patient's class (either 0 or 1)

### Sample Data Generation Using MIMIC-III

You can quickly test this reimplementation by creating a sample dataset from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) data using the `process_mimic_modified.py` script.  You will need to request access to [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/), a de-identified database containing information about clinical care of patients for 11 years of data, to be able to run this script.  If you do not wish to request access to the full data, you can freely download the [MIMIC-III](https://physionet.org/content/mimiciii-demo/1.4/) sample demo data and use it for exploratory benchmarks.  The `process_mimic_modified.py` script heavily borrows from the original [process_mimic.py](https://github.com/mp2893/retain/blob/master/process_mimic.py) script created by Edward Choi but is modified to output data in a format specified above. It outputs the necessary files to a user-specified directory and splits them into train and test by a user-specified ratio.

Example:

Run from the MIMIC-III directory. This will split data with 70% going to training and 30% to test:  

`python process_mimic_modified.py ADMISSIONS.csv DIAGNOSES_ICD.csv PATIENTS.csv data .7`

### License

Please review the [license](LICENSE), [notice](Notice.txt) and other [documents](docs/) before using the code in this repository or making a contribution to the repository

### Contributing

To contribute features, bug fixes, tests, examples, or documentation, please submit a pull request with a description of your proposed changes or additions.

Please include a brief description of your pull request when submitting code and ensure that your code follows the [Pep 8](https://www.python.org/dev/peps/pep-0008/) style guide.  To do this run `pip install black` and `black retain-keras` to reformat files within your copy of the code using the [black code formatter](https://github.com/psf/black).  The black code formatter is a PEP 8 compliant, opinionated formatter that reformats entire files in place.  You can also use the [autopep8 code formatter](https://packagecontrol.io/packages/AutoPEP8) within your IDE to ensure Pep 8 compliance.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


### References

1. Edward Choi, Mohammad Taha Bahadori, Joshua A. Kulas, Andy Schuetz, Walter F. Stewart, Jimeng Sun, 2016, RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism, In Proc. of Neural Information Processing Systems (NIPS) 2016, pp.3504-3512. https://github.com/mp2893/retain

2. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13).
