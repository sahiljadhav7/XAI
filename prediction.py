import numpy as np
import pandas as pd
import shap
import json
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import io
import shap
import base64

CKD_DATA_PATH = "static/models/chronic_kidney_disease/data/processed_kidney_disease.csv"
CKD_FEATURE_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"
]

CATEGORICAL_VALUE_MAP = {
    "rbc": {
        "normal": "normal",
        "abnormal": "abnormal",
    },
    "pc": {
        "normal": "normal",
        "abnormal": "abnormal",
    },
    "pcc": {
        "present": "present",
        "yes": "present",
        "notpresent": "notpresent",
        "not present": "notpresent",
        "no": "notpresent",
    },
    "ba": {
        "present": "present",
        "yes": "present",
        "notpresent": "notpresent",
        "not present": "notpresent",
        "no": "notpresent",
    },
    "htn": {"yes": "yes", "no": "no"},
    "dm": {"yes": "yes", "no": "no"},
    "cad": {"yes": "yes", "no": "no"},
    "appet": {"good": "good", "poor": "poor"},
    "pe": {"yes": "yes", "no": "no"},
    "ane": {"yes": "yes", "no": "no"},
}


def normalize_text(value):
    return str(value).strip().lower().replace("\t", "")


def normalize_numeric_value(value):
    if isinstance(value, list):
        return value[0] if value else np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return value

    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if match:
        number = match.group(0)
        return float(number) if "." in number else int(number)

    return np.nan


def normalize_input_data(data_dict):
    normalized = {}
    numeric_features = {
        "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wc", "rc"
    }

    for feature in CKD_FEATURE_COLUMNS:
        value = data_dict.get(feature, "")

        if feature in CATEGORICAL_VALUE_MAP:
            if pd.isna(value):
                normalized[feature] = np.nan
                continue

            key = normalize_text(value)
            if key in {"", "nan", "none"}:
                normalized[feature] = np.nan
                continue

            mapped_value = CATEGORICAL_VALUE_MAP[feature].get(key)
            if mapped_value is None:
                raise ValueError(f"Unsupported value '{value}' for {feature}")
            normalized[feature] = mapped_value
        elif feature in numeric_features:
            normalized[feature] = normalize_numeric_value(value)
        else:
            normalized[feature] = value

    return normalized

def get_preprocessor(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure 'classification' exists before modifying
    if 'classification' in data.columns:
        data['classification'] = data['classification'].replace(['ckd\t'], 'ckd')

    # Ensure 'id' exists before dropping it
    if 'id' in data.columns:
        data = data.drop(['id'], axis=1)

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    categorical_defaults = {
        col: data[col].mode()[0] for col in data.select_dtypes(include=['object']).columns
    }

    # Label Encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Drop classification column
    X = data.drop('classification', axis=1, errors='ignore')

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return scaler, label_encoders, X.columns, categorical_defaults  # Return feature names too


def preprocess_input_data(data_dict):
    """Preprocesses a single row of data in dictionary format."""

    try:
        # Load the scaler and encoders
        scaler, label_encoders, feature_columns, categorical_defaults = get_preprocessor(CKD_DATA_PATH)

        normalized_data = normalize_input_data(data_dict)
        print("Input Data:", normalized_data)

        # Convert dictionary to DataFrame
        instance_df = pd.DataFrame([normalized_data])

        # Ensure all required columns exist
        for col in feature_columns:
            if col not in instance_df.columns:
                instance_df[col] = 0  # Default for missing numerical values

        # Encode categorical columns
        for col in label_encoders:
            if col in instance_df.columns:
                instance_df[col] = instance_df[col].astype("object")
                if pd.isna(instance_df.at[0, col]):
                    instance_df.at[0, col] = categorical_defaults[col]
                value = instance_df[col].astype(str)
                if value.iloc[0] not in label_encoders[col].classes_:
                    raise ValueError(f"Unsupported categorical value '{value.iloc[0]}' for {col}")
                instance_df[col] = label_encoders[col].transform(value)

        # Fill missing values with median (for safety)
        instance_df.fillna(instance_df.median(numeric_only=True), inplace=True)

        # Ensure columns are in correct order
        instance_df = instance_df[feature_columns]

        # Standardize numerical data
        scaled_instance = scaler.transform(instance_df)

        print("Preprocessed Data:", type(scaled_instance))
        return scaled_instance

    except (KeyError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None
    
    
def predict_explain(scaled_instance, rf_model, explainer, columns):
    """Predicts the class and generates SHAP values for a single instance.

    Args:
      scaled_instance: A NumPy array representing a single scaled instance.
      rf_model: The trained RandomForestClassifier model.
      explainer: The SHAP explainer object.
      X_test_df: The DataFrame of test features.

    Returns:
        A tuple containing the prediction, force plot HTML, and SHAP values.
    """

    try:
        if len(scaled_instance.shape) == 1:
            scaled_instance = scaled_instance.reshape(1, -1)
            print(scaled_instance.shape)
        else:
            print("scaled_instance is already a 2D array with shape:", scaled_instance.shape)
        # Make predictions
        predicted_class = rf_model.predict(scaled_instance)[0]
        predicted_probs = rf_model.predict_proba(scaled_instance)[0]
        # print(predicted_class)
        # print(predicted_probs)
        
        

        # Extract SHAP values
        shap_values_single = explainer.shap_values(scaled_instance) 
        print(f"Shape of shap_values_single: {shap_values_single.shape}") # Use the explainer directly
        shap_values_for_class = shap_values_single[0, :, predicted_class]
        print(f"Shape of shap_values_for_class: {shap_values_for_class.shape}")
        
        
        return predicted_class,predicted_probs[0],shap_values_for_class,explainer

    except Exception as e:
        print(f"An error occurred during prediction or explanation: {e}")
        return None, None, None



def get_explainer(rf_model, X_test):
    """
    Generates and returns a SHAP explainer for the given model and data.

    Args:
        rf_model: The trained random forest model.
        X_test:  The test data (features).

    Returns:
        A SHAP explainer object.
    """
    try:
      explainer = shap.TreeExplainer(rf_model)
      return explainer
    except Exception as e:
      print(f"Error creating explainer: {e}")
      return None
