# Twitter Bot Detection - Exploring Traditional, Ensemble and Multimodal Models

This repository contains a fully containerised and reproducible machine learning workflow for Twitter bot detection. It provides modular preprocessing, multiple classical ML models (XGBoost, Random Forest, Logistic Regression, LDA, QDA, Naive Bayes), and an optional multimodal deep-learning model. All scripts are executed inside a controlled Docker environment, with a **Makefile** providing a unified command interface.

## Contributors:
| Name                                       | Matriculation No. |
| - | -- |
| Roderich Suwandi Leee                        | A0259051H      |
| Yoong Jun Han                                |  A0252837X   |
| Ho Xin En Jolene                           |   A0257596J      |
| Tay Xin Ru Rena                             | A0256841Y        |
| Choy Qi Hui                                  |    A0259587E   |


## Directory Structure

```
├── data/
│   ├── raw/                 # Original dataset
│   ├── interim/             # Intermediate preprocessing outputs
│   ├── cleaned/             # Fully processed data
│   └── image_cache/         # Cached images/features for multimodal model (if generated)
│
├── models/                  # Model artefacts generated after training
│   ├── randomforest/
│   ├── traditional/
│   └── xgboost/
│
├── reports/                 # Evaluation outputs (plots, metrics)
│   ├── gaussiannb/
│   ├── lda/
│   ├── logreg/
│   ├── qda/
│   ├── randomforest/
│   └── xgboost/
│
├── notebooks/               # EDA notebooks
|   ├── archived/            # old notebooks not used in report
│   └── eda.ipynb            # main notebook used in report for EDA 
│
├── src/
│   ├── models/                     # scripts to train,tune and eval models
│   │   ├── xgboost_model.py
│   │   ├── randomforest_model.py
│   │   ├── traditional_ml_model.py
│   │   └── multimodal_model.py
│   ├── preprocessing/              # scripts used to preprocess raw data
│   │   ├── preprocess.py
│   │   └── preprocess_multimodal.py
│   └── viz/                         # helper utility scripts for plotting graphs, confusion matrix
│       ├── visualize_multimodal.py
│       └── visualize_utility.py
│
├── docker-compose.yml
├── Dockerfile
├── Makefile                         # Shortcut commands for running scripts/notebooks
└── requirements.txt

````

## Installation
Ensure you have the following installed:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/standalone/) (if running on Linux)

1. Clone the repo
   ```bash
   git clone https://github.com/yjh-jy/dsa4263-twitter-bot-detection
   ```

2. Navigate to directory
   ```bash
   cd dsa4263-twitter-bot-detection
   ```

## How to Run

Below is the recommended workflow for executing the project end-to-end.



### **Step 1 — Build the Docker Environment**

```bash
make build
````

Rebuild from scratch:

```bash
make build-nocache
```



### **Step 2 — (Optional) Launch JupyterLab for EDA**

```bash
make notebook
```

This starts JupyterLab at:

```
http://localhost:8888
```


### **Step 3 — Run Preprocessing**

Preprocessing scripts are located in `src/preprocessing/`.

Run individual tasks:

```bash
make prep
make prep-multimodal
```

Outputs are written to:

```
data/cleaned/
data/cleaned/multimodal
```

### **Step 4 — Train, Tune and Evaluate Models**

All the following commands run inside the `runtime` container.

#### XGBoost

```bash
make xgb # evaluation only
```

```bash
make xgb-retrain # for retraining xgboost model
```

#### Random Forest

```bash
make rf
```

#### Traditional ML Models

(Logistic Regression, LDA, QDA, Naive Bayes)

```bash
make trad
```

#### Run all the main models in sequence
```bash
make all-models
```

#### (Optional) Multimodal Model

```bash
make multimodal
```
> [!WARNING] Note that this takes very long to complete as gpu support is not provided in this containerisation to accomodate for all systems (MacOS, Windows, Linux)
Trained artefacts will appear under:

```
models/<model_name>/
reports/<model_name>/
```


### **Step 5 — Use Developer Utilities**

Open an interactive shell:

```bash
make bash
```

Stop all containers:

```bash
make down
```

Delete all containers and volumes (including generated models, reports):

```bash
make clean
```

## Notes on Persistence

* Source code and datasets are bind-mounted (`./data:/app/data`).
* Model outputs, reports, and image caches are stored in **Docker named volumes**, ensuring persistence across container restarts and rebuilds.
* These volumes are deleted only when running:

  ```bash
  make clean
  ```

## Summary of Main Commands

| Task                                       | Command           |
| - | -- |
| Build Docker image                         | `make build`      |
| Run JupyterLab                             |  `make notebook`   |
| Train, Tune, Eval XGBoost                   | `make xgb`        |
| Train, Tune, Eval Random Forest             | `make rf`         |
| Train, Tune, Eval classical ML models       | `make trad`       |
| Train, Tune, Eval all the main models       | `make all-models`       |
| Train, Tune, Eval multimodal model          | `make multimodal` |
| Shell into container                          | `make bash`       |
| Stop containers                               | `make down`       |
| Remove all volumes & containers               | `make clean`      |
