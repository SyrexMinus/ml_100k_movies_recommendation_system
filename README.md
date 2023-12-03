Makar Shevchenko | m.shevchenko@innopolis.university | B20-DS-01

# movies_recommendation_system | | PMLDL course | Assignment 2

Matrix factorization based recommendation system for movies in MovieLens 100K (ml-100k) dataset. A project for F23 
Practical Machine Learning and Deep Learning course in Innopolis University.

In this work I explored, preprocessed MovieLens 100K dataset and applied it for training and evaluation of 
matrix factorization based recommendation model LightFM.

## Navigation in the Repo

The task that I solved is described in `references/task_description.md`

The description of work procedure is in `reports/final_report.pdf`

The notebooks on data exploration & preprocessing and on model training & evaluation are in `notebooks`

The model might be reevaluated with `benchmark/evaluate.py`

The list of resources used is listed in `references/references.bib`

The weights of the trained model are in `models/model_ckpt.pkl`

The `data` directory is used for storage of data for scripts. `external` subdirectory contain data from third party
sources, `inernal` - intermediate data that has been transformed, and `raw` - the original, immutable data.

## Usage

Before executing any script make sure that your Python is of version 3.10.3 or above. All the commands should be 
executed from the repo root. Notebooks should be executed from `notebooks` directory.
The commands are tested on MacOS 13.5.2.

If you face issues with imports, install the missing packages

### Data Exploration & Preprocessing

1. `cd notebooks`
2. `jupyter notebook`
3. Open `1.0-data-exploration-and-preprocessing.ipynb` in GUI
4. Run all the code cells

The script will store raw dataset in `data/raw` and train and intermediate files in `data/interim`

### Model Training

0. Complete Data Exploration & Preprocessing
1. `cd notebooks`
2. `jupyter notebook`
3. Open `2.0-model-training-and-visualization.ipynb` in GUI
4. Run all the code cells

The script will train the LightFM model, store the trained weights in `models` and the evaluation data in 
`benchmark/data/eval_data.pkl`.

### Predicting

0. Go through Model Training
1. Fill `user_ids` in the last code cell with ids of the users you want to recommend to
