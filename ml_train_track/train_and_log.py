import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing

import mlflow
import mlflow.sklearn
import logging

# Configure logging for better visibility of MLflow operations
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def train_and_log_model(data, features:list, test_size=0.2):
    """
    Trains a Linear Regression model, evaluates it, and logs the experiment details
    to MLflow.
    """
    mlflow.set_experiment("Estimate Housing Price")
    with mlflow.start_run():

        # 1. Load the Dataset
        # For newer scikit-learn versions, you might need to adjust this part.
        # Example: from sklearn.datasets import fetch_california_housing
        # housing = fetch_california_housing()
        housing = data
        X = pd.DataFrame(housing.data, columns=features)
        y = pd.Series(housing.target)

        # 2. Split Data into Training and Testing Sets
        random_state_nr = 42  # For reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state_nr)

        # 3. Initialize and Train the Model
        # We'll use a simple Linear Regression for this example.
        model = RandomForestRegressor() # other Model: LinearRegression()
        model.fit(X_train, y_train)

        # 4. Make Predictions
        predictions = model.predict(X_test)

        # 5. Evaluate the Model
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Print metrics for immediate feedback
        print(f"  MSE: {mse:.3f}")
        print(f"  RMSE: {mae:.3f}")
        print(f"  R2: {r2:.3f}")

        # 6. Log Parameters to MLflow
        # Even though LinearRegression doesn't have 'alpha' or 'l1_ratio' directly,
        # we can log them as example parameters to demonstrate the functionality.
        # If you were using ElasticNet, these would be actual model parameters.
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", features)
        mlflow.log_param("random_state", random_state_nr)
        mlflow.log_param("model_type", "RandomForestRegressor")

        # 7. Log Metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # 8. Log the Model to MLflow
        # This saves the model in a format that MLflow can easily deploy.
        mlflow.sklearn.log_model(model, "random_regression_model")

        # You can also set a tag for the run
        mlflow.set_tag("experiment_name", "housing Housing Price Prediction")

        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    print("Starting MLflow experiment for Random Forest Regression on Housing dataset...")
    # You can call the function with different parameters to simulate multiple runs
    # and compare them in the MLflow UI.
    data = fetch_california_housing(as_frame=True)
    train_and_log_model(data,data.feature_names)
    train_and_log_model(data,data.feature_names, test_size=0.3)
    train_and_log_model(data,["Longitude","Latitude"])
    print("\nMLflow experiments complete. To view results, run 'mlflow ui' in your terminal.")
    print("If there is an error with socket, it is possible to use a different port by adding --port <port_number> to the command above.")