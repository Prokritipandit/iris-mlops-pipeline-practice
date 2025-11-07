import os
import joblib
import pandas as pd
import pytest
from feast import FeatureStore
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- FIXTURES: Re-usable setup code for tests ---

@pytest.fixture(scope="session")
def feast_store():
    """Connects to the Feast feature store."""
    # Use the known absolute path from our previous work
    fs = FeatureStore(repo_path="/home/jupyter/feature_repo/feature_repo")
    return fs

@pytest.fixture(scope="session")
def dvc_pulled_model():
    """
    Runs 'dvc pull' and loads the model.
    This assumes DVC is set up and will be run in the CI environment.
    """
    # Define absolute paths
    model_path = "/home/jupyter/models/model.joblib"
    encoder_path = "/home/jupyter/models/label_encoder.joblib"

    # Run DVC pull (this will be handled by CI, but good for local testing)
    os.system("dvc pull models/model.joblib models/label_encoder.joblib -f")
    
    # Load the files
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    
    return model, le

@pytest.fixture(scope="session")
def evaluation_data(feast_store):
    """
    Fetches the same historical feature data used for training,
    which we will use as our test set.
    """
    # Load the ground truth data
    ground_truth_path = "/home/jupyter/data/iris_ground_truth.csv"
    ground_truth = pd.read_csv(ground_truth_path)
    ground_truth['event_timestamp'] = pd.to_datetime(ground_truth['event_timestamp'])
    
    # Define features
    feature_names = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]
    
    # Get historical features
    training_df = feast_store.get_historical_features(
        entity_df=ground_truth,
        features=feature_names
    ).to_df()
    
    return training_df

# --- TESTS ---

def test_data_validation(evaluation_data):
    """
    Validates the data fetched from the feature store.
    """
    df = evaluation_data
    
    # 1. Check for non-nulls
    assert df.isnull().sum().sum() == 0, "Found NaN values in feature data"
    
    # 2. Check for expected columns
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_cols.issubset(set(df.columns)), "Data is missing expected columns"
    
    # 3. Check for reasonable data ranges (example)
    assert df['sepal_length'].min() > 0, "Sepal length should be positive"
    assert df['petal_length'].min() > 0, "Petal length should be positive"

def test_model_performance(dvc_pulled_model, evaluation_data):
    """
    Tests that the model achieves a minimum accuracy on the test set.
    """
    model, le = dvc_pulled_model
    df = evaluation_data
    
    # Define feature and target
    X_test = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test_raw = df["species"]
    
    # Encode target
    y_test = le.transform(y_test_raw)
    
    # Run prediction
    y_pred = model.predict(X_test)
    
    # Test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    # Set our minimum performance threshold
    assert accuracy > 0.90, f"Model accuracy {accuracy} is below the 0.90 threshold"

def test_online_prediction_sanity(feast_store, dvc_pulled_model):
    """
    Runs a sanity check on a single online prediction.
    """
    fs = feast_store
    model, le = dvc_pulled_model
    
    # 1. Fetch online features for a known ID
    entity_rows = [{"iris_id": 1001}]
    feature_names = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]
    
    online_features_dict = fs.get_online_features(
        features=feature_names,
        entity_rows=entity_rows
    ).to_dict()
    
    features_df = pd.DataFrame.from_dict(online_features_dict)
    
    # 2. Run prediction
    X_inference = features_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    prediction_encoded = model.predict(X_inference)
    prediction_label = le.inverse_transform(prediction_encoded)
    
    # 3. Assert a valid result
    assert prediction_label[0] in {"setosa", "versicolor", "virginica"}
    print(f"Online prediction for 1001: {prediction_label[0]}")
