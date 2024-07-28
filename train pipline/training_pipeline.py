import dagshub
import mlflow
from training import main

# Initialize DagsHub
dagshub.init(repo_owner='yasmine-jammeli', repo_name='mlflow_project', mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/yasmine-jammeli/mlflow_project.mlflow')

# Execute the main training process
main()
