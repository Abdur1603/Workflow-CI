import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub
import os
import shutil
matplotlib.use('Agg')

REPO_OWNER = "Abdur1603"
REPO_NAME = "Eksperimen_SML"

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

    try:
        df = pd.read_csv('telco_customer_churn_preprocessing/telco_churn_clean.csv')
    except FileNotFoundError:
        print("CRITICAL: Dataset not found.")
        exit(1)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"--> Run ID: {run_id}")

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        print("Uploading model to DagsHub (Online Backup)...")
        mlflow.sklearn.log_model(model, "model")

        local_model_path = "model_output"
        
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        print(f"Saving model locally to '{local_model_path}' for Docker Build...")
        mlflow.sklearn.save_model(model, local_model_path)
        print("Local save success.")

        with open("last_run_id.txt", "w") as f:
            f.write(run_id)

if __name__ == "__main__":
    main()