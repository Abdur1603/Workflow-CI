import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train():
    data_path = "namadataset_preprocessing" 
    X_train, X_test, y_train, y_test = load_data(data_path)

    mlflow.set_experiment("CI_Experiment")

    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50], 'max_depth': [5]}

    print("Starting Grid Search...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run():
        mlflow.log_params(best_params)
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(best_model, "model")
        print(f"Model Trained. Accuracy: {acc}")
        
        plt.figure()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

if __name__ == "__main__":
    train()