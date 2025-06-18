import mlflow 
import mlflow.sklearn
import pandas as pd
import mlflow.sklearn   
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 

import dagshub
dagshub.init(repo_owner='mjcode14', repo_name='mlflow-dagshub-demo', mlflow=True)
import pandas as pd

import mlflow.sklearn

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import seaborn as sns

mlflow.set_tracking_uri('https://dagshub.com/mjcode14/mlflow-dagshub-demo.mlflow')

# Load the Iris dataset

iris = load_iris()

X = iris.data

y = iris.target

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier

max_depth = 8

n_estimators = 10

mlflow.set_experiment("iris_dt")

#apply miflow to train

# with mlflow.start_run(run_name="pk_exp_with_confusion_matrix_log_artifact"):
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    #Create confusion matrix
    

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)
    cm  = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    # mlflow.sklearn.log_model(dt, "decision Tree")
    mlflow.set_tag('author', 'Mayur')
    mlflow.set_tag('model', 'decision tree')
    print("accuracy",accuracy)