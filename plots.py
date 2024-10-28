import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np

# Create a directory to save the plots
plot_dir = "C:/Users/pc/Desktop/Nesrin master/plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Load dataset
combined_csv_path = 'C:/Users/pc/Desktop/Nesrin master/combined_features_with_more_Time_feauture.csv'
data = pd.read_csv(combined_csv_path)

# Split data into features and target (assuming the last column is the target/labels)
X = data.iloc[:, :-1]  # All columns except the last one as features
y = data.iloc[:, -1]    # The last column as labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(),
    "LightGBM": lgb.LGBMClassifier()
}

# Function to plot confusion matrix and learning curves
def plot_confusion_matrix_and_learning_curve(model, model_name):
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{model_name} - Confusion Matrix')
    plt.savefig(f'{plot_dir}/{model_name}_confusion_matrix.png')
    plt.close()

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
    plt.title(f'{model_name} - Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'{plot_dir}/{model_name}_learning_curve.png')
    plt.close()

# Train and plot for each model
for model_name, model in models.items():
    plot_confusion_matrix_and_learning_curve(model, model_name)

print(f"All plots have been saved to: {plot_dir}")
