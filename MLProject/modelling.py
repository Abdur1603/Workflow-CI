import pandas as pd
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')

def load_data(data_path):
    print("Loading data...")
    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_save(X_train, X_test, y_train, y_test):
    print("Training Model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")
    if os.path.exists("confusion_matrix.png"): os.remove("confusion_matrix.png")

    local_path = "model_output"
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    
    print(f"Saving model locally to '{local_path}' for Docker...")
    mlflow.sklearn.save_model(rf, local_path)
    print("Model saved successfully.")

    mlflow.sklearn.log_model(rf, "model")

def main():
    mlflow.sklearn.autolog()

    data_folder = "telco_customer_churn_preprocessing"
    if not os.path.exists(data_folder):
        data_folder = "."

    X_train, X_test, y_train, y_test = load_data(data_folder)

    if mlflow.active_run():
        print("Active Run detected (CI/CD environment). Using existing run.")
        train_and_save(X_train, X_test, y_train, y_test)
    else:
        print("No Active Run detected (Local environment). Starting new run.")
        with mlflow.start_run(run_name="Manual_Run"):
            train_and_save(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()