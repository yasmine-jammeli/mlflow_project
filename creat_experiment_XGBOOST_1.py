import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

combined_df = pd.read_csv('C:/Users/pc/Desktop/Nesrin master/combined_features_with_more_feauture.csv')

X = combined_df.iloc[:, :-1].values
y = combined_df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def feature_selection(X_train, y_train):
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

X_train_selected, X_test_selected, selector = feature_selection(X_train, y_train)

# Define hyperparameter configurations
param_configs = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'lambda': 0, 'alpha': 0},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.1, 'lambda': 0.1, 'alpha': 0.1},
    {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.2, 'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 0.2, 'lambda': 1.0, 'alpha': 1.0},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.1, 'lambda': 0.1, 'alpha': 0.1},
    {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.2, 'subsample': 0.9, 'colsample_bytree': 1.0, 'gamma': 0.2, 'lambda': 0.5, 'alpha': 0.5},
    {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 0.8, 'gamma': 0, 'lambda': 1.0, 'alpha': 0.5},
    {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.9, 'gamma': 0.1, 'lambda': 0.1, 'alpha': 0.1}
]

# Run experiments
for idx, params in enumerate(param_configs):
    experiment_name = f"experiment_{idx+1}"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        xgb_model = XGBClassifier(eval_metric='mlogloss', **params)
        xgb_model.fit(X_train_selected, y_train)
        
        y_pred = xgb_model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(xgb_model, "model")

        print(f"Experiment {idx+1} Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_report(y_test, y_pred))
       
        ConfusionMatrixDisplay(conf_matrix).plot()
        plt.savefig(f"confusion_matrix_experiment_{idx+1}.png")
        plt.show()
