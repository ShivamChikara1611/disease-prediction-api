import os
import sys
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_file, test_file):
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        logging.info("Data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logging.error("Error loading data: %s", str(e))
        sys.exit(1)

def preprocess_data(data):
    try:
        if 'prognosis' not in data.columns:
            raise ValueError("Column 'prognosis' not found in data.")
        X = data.drop('prognosis', axis=1)
        y = data['prognosis']
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        sys.exit(1)

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info("Model evaluation complete.")
        return accuracy, report
    except Exception as e:
        logging.error("Error evaluating model: %s", str(e))
        sys.exit(1)

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute file paths
    train_file = os.path.join(current_dir, "..", "data", "Training.csv")
    test_file = os.path.join(current_dir, "..", "data", "Testing.csv")
    
    # Load the data
    train_data, test_data = load_data(train_file, test_file)
    
    # Preprocess the data
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    logging.info("Model Accuracy: %.4f", accuracy)
    logging.info("Classification Report:\n%s", report)
    
    # Save the model
    model_path = os.path.join(current_dir, "..", "models", "disease_prediction_model.pkl")
    try:
        joblib.dump(model, model_path)
        logging.info("Model saved at %s", model_path)
    except Exception as e:
        logging.error("Error saving model: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()