import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference

# Just creating a synthetic dataset to be used for testing
@pytest.fixture
def data():
    return pd.DataFrame({
        "workclass": ["Private", "Private", "State-gov"],
        "education": ["Bachelors", "Masters", "HS-grad"],
        "marital-status": ["Married", "Single", "Divorced"],
        "occupation": ["Exec", "Prof", "Repair"],
        "relationship": ["Husband", "Wife", "Own-child"],
        "race": ["White", "Black", "Asian"],
        "sex": ["Male", "Female", "Male"],
        "native-country": ["US", "US", "Mexico"],
        "salary": [">50K", "<=50K", "<=50K"]
    })

def test_process_data_count(data):
    """ Test that processing does not lose any rows of data. """
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=["workclass", "education", "marital-status", "occupation", 
                              "relationship", "race", "sex", "native-country"], 
        label="salary", 
        training=True
    )
    
    # Check that we still have 3 rows
    assert len(X) == 3
    assert len(y) == 3


def test_train_model_exists(data):
    """ Test that the training function returns a model object. """
    X, y, _, _ = process_data(data, categorical_features=["workclass", "education", "marital-status", "occupation", 
                              "relationship", "race", "sex", "native-country"], label="salary", training=True)
    model = train_model(X, y)
    
    # Check that model is not None
    assert model is not None


def test_inference_binary(data):
    """ Test that inferences result in 0 or 1, not some other number. """
    X, y, _, _ = process_data(data,categorical_features=["workclass", "education", "marital-status", "occupation", 
                              "relationship", "race", "sex", "native-country"], label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    
    # Check that every prediction is either 0 or 1
    assert np.all((preds == 0) | (preds == 1))
