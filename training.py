import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Load data
combined_df = pd.read_csv('C:/Users/pc/Desktop/Nesrin master/combined_features_with_more_feauture.csv')

# Split data
X = combined_df.iloc[:, :-1].values
y = combined_df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
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

# Hyperparameter Tuning with Randomized Search
def randomized_search_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(eval_metric='mlogloss')
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'lambda': [0, 0.1, 1.0],
        'alpha': [0, 0.1, 1.0]  # Include alpha for L1 regularization
    }
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, n_iter=50, cv=3, n_jobs=-1, scoring='accuracy', verbose=1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search

# Run Randomized Search
random_search_result = randomized_search_xgboost(X_train_selected, y_train)
best_xgb_model = random_search_result.best_estimator_

# Evaluation
y_pred = best_xgb_model.predict(X_test_selected)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.show()

# Logging with MLflow
with mlflow.start_run():
    mlflow.log_params(random_search_result.best_params_)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.sklearn.log_model(best_xgb_model, "model")