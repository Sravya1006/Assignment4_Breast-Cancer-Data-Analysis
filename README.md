# Assignment4_Breast-Cancer-Data-Analysis

## Overview

This project aims to build a predictive model for breast cancer classification using an Artificial Neural Network (ANN). The project is implemented in Python and leverages a variety of data science libraries for data preprocessing, model building, and deployment through a Streamlit web application.

## Project Structure

- `app.py`: This is the main script for the Streamlit app. It provides an interactive interface for users to upload breast cancer data, preprocess it, and get predictions using the trained ANN model.
- `notebook.ipynb`: A Jupyter notebook that documents the data exploration, preprocessing, and feature selection process.
- `prediction.ipynb`: A Jupyter notebook detailing the model building, training, and evaluation process.
- `requirements.txt`: A file listing all the necessary Python packages and their versions required to run the project.

## Requirements

To set up the project locally, you'll need the following Python packages, as specified in `requirements.txt`:

- `tensorflow==2.16.1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `scikeras`

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## Project Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Jupyter Notebooks

- Use `notebook.ipynb` to explore and preprocess the breast cancer dataset.
- Use `prediction.ipynb` to build, train, and evaluate the ANN model.

### Streamlit App

1. Run the Streamlit app using:
   ```bash
   streamlit run app.py
   ```

2. Interact with the app in your browser. You can upload new data, preprocess it, and get predictions from the ANN model.

## Data

The breast cancer dataset used in this project can be obtained from the UCI Machine Learning Repository, Kaggle, or directly from `sklearn.datasets`.

## Feature Selection

The project employs feature selection techniques using `SelectKBest` from `sklearn.feature_selection` to enhance model performance by selecting the most relevant features.

## Model Tuning

Grid Search Cross-Validation (`GridSearchCV`) is used to optimize the hyperparameters of the ANN model, ensuring the best possible performance on the dataset.

## Deployment

The project is deployed locally using Streamlit, allowing for an interactive user interface. For deployment on a remote server, consider using platforms like Heroku, AWS, or Google Cloud.

## Version Control

The project is version controlled using Git. Regular commits are made to track changes, and the repository is hosted on GitHub.
