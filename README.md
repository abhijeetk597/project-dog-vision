Project-dog-vision
==============================

Dog_breed_classification - Deep Learning / Transfer Learning Project including deployment on Streamlit

🚀 Excited to share my journey diving into the world of Deep Learning with my first project, 'Dog Vision'! 🐶
Using TensorFlow, I delved into the complexities of neural networks and brought this project to life, complete with deployment on Streamlit Cloud.

🔍 When I was working in the Bank, I was always wondered about how CTS machine detects cheque no and micr code, or how CKYC scanner detects the type of ID proof. Now, as I immerse myself in Deep Learning, those mysteries are unraveling.

This project uses data from Kaggle's 'Dog Breed Identification' competition with over 120 breeds and 10k+ images in both training and test sets.

📈 The execution plan was a two-phase approach:
**First Go:**
1. Prepping images and labels into Tensors
2. Experimenting with models like mobilenet_v2
3. Training and scoring predictions on Kaggle
**Second Go:**
1. Augmenting data by flipping images
2. Employing another model, resnet_v2_50
3. Training, scoring, and validating predictions

🌐 Then came deployment:
1. Crafted an interactive app on Streamlit for breed predictions
2. Published the project on GitHub
3. Successfully deployed the app on Streamlit Cloud

🔑 Key Learnings:
- Deep dive into TensorFlow and Neural Networks
- Implementing Transfer Learning with diverse pre-trained models
- Harnessing GPU power for training and predictions
- Seamlessly deploying models using Streamlit

Streamlit App- https://lnkd.in/dAqtsXPy

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
