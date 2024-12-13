import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def load_data(self):
        """Load the processed features."""
        try:
            processed_dir = "/home/x-raf/Projects/mlops/mlops-pipeline/training-model-pipeline/data/processed"
            files = os.listdir(processed_dir)
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(processed_dir, x)))
            data_path = os.path.join(processed_dir, latest_file)
            
            df = pd.read_csv(data_path)
            logger.info(f"Training data loaded from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise

    def prepare_data(self, df):
        """Prepare data for training."""
        try:
            # Separate features and target
            X = df.drop('target', axis=1)
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            logger.info("Data prepared for training")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, X_train, y_train):
        """Train the model."""
        try:
            # Start MLflow run
            with mlflow.start_run():
                # Log model parameters
                mlflow.log_params({
                    "n_estimators": self.model.n_estimators,
                    "max_depth": self.model.max_depth
                })
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                logger.info("Model training completed")
                return self.model
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self, model):
        """Save the trained model."""
        try:
            # Create models directory if it doesn't exist
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_dir, f"model_{timestamp}.pkl")
            mlflow.sklearn.save_model(model, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("model_retraining")
        
        trainer = ModelTrainer()
        
        # Load and prepare data
        df = trainer.load_data()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # Train model
        model = trainer.train_model(X_train, y_train)
        
        # Save model
        model_path = trainer.save_model(model)
        
        logger.info("Model training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 