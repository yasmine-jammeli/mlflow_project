import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Load data
combined_df = pd.read_csv('C:/Users/pc/Desktop/Nesrin master/combined_features_with_more_feauture.csv')

# Split data
X = combined_df.iloc[:, :-1].values
y = combined_df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection
def feature_selection(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

X_train_selected, X_test_selected, selector = feature_selection(X_train, y_train)

# Define hyperparameter configurations
param_configs = [
    {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 10, 'min_samples_leaf': 4},
    {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 6},
    {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 20, 'min_samples_leaf': 8},
    {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 25, 'min_samples_leaf': 10},
    {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 30, 'min_samples_leaf': 12}
]

# Run experiments
for idx, params in enumerate(param_configs):
    experiment_name = f"DecisionTree_Exp_{idx+1}_Criterion_{params['criterion']}_Depth_{params.get('max_depth', 'None')}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Create and train model
        dt_model = DecisionTreeClassifier(**params)
        dt_model.fit(X_train_selected, y_train)
        
        # Predict and evaluate
        y_pred = dt_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model
        mlflow.sklearn.log_model(dt_model, "model")

        # Print results
        print(f"Experiment {idx+1} Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Plot and save the confusion matrix
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.savefig(f"confusion_matrix_decision_tree_experiment_{idx+1}.png")
        plt.show()
