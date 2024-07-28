import dagshub
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow.sklearn

# Initialize DagsHub
dagshub.init(repo_owner='yasmine-jammeli', repo_name='mlflow_project', mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/yasmine-jammeli/mlflow_project.mlflow')

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
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

X_train_selected, X_test_selected, selector = feature_selection(X_train, y_train)

# Define XGBoost hyperparameter configurations
xgb_param_configs = [
    {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'lambda': 0.1, 'alpha': 0.1},
    {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.2, 'lambda': 0.2, 'alpha': 0.2},
    {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 0.3, 'lambda': 0.3, 'alpha': 0.3},
    {'n_estimators': 250, 'max_depth': 10, 'learning_rate': 0.3, 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.4, 'lambda': 0.4, 'alpha': 0.4},
    {'n_estimators': 300, 'max_depth': 12, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 1.0, 'gamma': 0.5, 'lambda': 0.5, 'alpha': 0.5},
    {'n_estimators': 350, 'max_depth': 15, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 0.8, 'gamma': 0.6, 'lambda': 0.6, 'alpha': 0.6},
    {'n_estimators': 400, 'max_depth': 20, 'learning_rate': 0.3, 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.7, 'lambda': 0.7, 'alpha': 0.7}
]

# Run XGBoost experiments
for idx, params in enumerate(xgb_param_configs):
    experiment_name = f"XGBoost_Exp_{idx+1}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Create and train model
        xgb_model = XGBClassifier(eval_metric='mlogloss', **params)
        xgb_model.fit(X_train_selected, y_train)
        
        # Predict and evaluate
        y_pred = xgb_model.predict(X_test_selected)
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
        mlflow.sklearn.log_model(xgb_model, "model")

        # Register the model
        model_name = f"XGBoost_Model_{idx+1}"
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)

        # Print results
        print(f"Experiment {idx+1} Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_report(y_test, y_pred))

# Define DecisionTree hyperparameter configurations
dt_param_configs = [
    {'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2},
    {'criterion': 'gini', 'max_depth': 7, 'min_samples_split': 10, 'min_samples_leaf': 4},
    {'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 15, 'min_samples_leaf': 6},
    {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 20, 'min_samples_leaf': 8},
    {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 25, 'min_samples_leaf': 10},
    {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 30, 'min_samples_leaf': 12}
]

# Run DecisionTree experiments
for idx, params in enumerate(dt_param_configs):
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

        # Register the model
        model_name = f"DecisionTree_Model_{idx+1}"
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)

        # Print results
        print(f"Experiment {idx+1} Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_report(y_test, y_pred))
