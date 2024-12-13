import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def load_test_data(self):
        """Load the test data."""
        try:
            processed_dir = "/home/x-raf/Projects/mlops/mlops-pipeline/training-model-pipeline/data/processed"
            files = os.listdir(processed_dir)
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_dir, x)))
            data_path = os.path.join(processed_dir, latest_file)
            
            df = pd.read_csv(data_path)
            X_test = df.drop('target', axis=1)
            y_test = df['target']
            
            logger.info(f"Test data loaded from {data_path}")
            return X_test, y_test
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise

    def load_model(self):
        """Load the latest trained model."""
        try:
            models_dir = "/home/x-raf/Projects/mlops/mlops-pipeline/training-model-pipeline/models"
            files = os.listdir(models_dir)
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            model_path = os.path.join(models_dir, latest_file)
            
            model = mlflow.sklearn.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Log metrics with MLflow
            with mlflow.start_run():
                mlflow.log_metrics(self.metrics)
            
            logger.info("Model evaluation completed")
            return self.metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_metrics(self):
        """Save evaluation metrics."""
        try:
            # Create metrics directory if it doesn't exist
            metrics_dir = "metrics"
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            logger.info(f"Metrics saved to {metrics_path}")
            return metrics_path
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

def main():
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("model_evaluation")
        
        evaluator = ModelEvaluator()
        
        # Load test data and model
        X_test, y_test = evaluator.load_test_data()
        model = evaluator.load_model()
        
        # Evaluate model
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        # Save metrics
        metrics_path = evaluator.save_metrics()
        
        logger.info("Model evaluation pipeline completed successfully")
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 