import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
matplotlib.use('Agg')

REPO_OWNER = "Abdur1603"
REPO_NAME = "Eksperimen_SML"

def load_data(data_path):
    print("Loading data...")
    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan di path yang ditentukan.")
        return None, None, None, None

def main():
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        try:
            dagshub.auth.add_app_token(token)
        except Exception:
            pass
    
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
    mlflow.set_experiment("Telco_Churn_Docker_Build")

    data_folder = "telco_customer_churn_preprocessing" 
    
    if not os.path.exists(data_folder):
        data_folder = "."
    
    X_train, X_test, y_train, y_test = load_data(data_folder)
    
    if X_train is None:
        exit(1)

    print("Starting Hyperparameter Tuning...")
    
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best Params found: {best_params}")

    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run(run_name="RandomForest_Tuned_CI") as run:
        run_id = run.info.run_id
        print(f"--> Run ID: {run_id}")
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "Random Forest (Tuned)")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Metrics Logged -> Acc: {acc:.4f}, F1: {f1:.4f}")

        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Acc: {acc:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10,6))
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X_train.columns
            
            plt.title("Feature Importances")
            plt.bar(range(X_train.shape[1]), importances[indices], align="center")
            plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            mlflow.log_artifact("feature_importance.png")

        print("Uploading model to DagsHub...")
        mlflow.sklearn.log_model(best_model, "model")

        local_model_path = "model_output"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        print(f"Saving model locally to '{local_model_path}' for Docker Build...")
        mlflow.sklearn.save_model(best_model, local_model_path)
        print("Local save success.")

        with open("last_run_id.txt", "w") as f:
            f.write(run_id)

    for f in ["confusion_matrix.png", "feature_importance.png"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()