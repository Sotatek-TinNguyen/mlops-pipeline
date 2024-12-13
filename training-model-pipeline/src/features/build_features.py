import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, data_path):
        """Load data from the latest raw data file."""
        try:
            # Get the latest file from raw data directory
            raw_data_dir = "/home/x-raf/Projects/mlops/mlops-pipeline/training-model-pipeline/data/raw"
            files = os.listdir(raw_data_dir)
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(raw_data_dir, x)))
            data_path = os.path.join(raw_data_dir, latest_file)
            
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_features(self, df):
        """Create new features from the input data."""
        try:
            # Example feature engineering steps
            # 1. Create interaction features
            df['interaction'] = df['feature1'] * df['feature2']
            
            # 2. Create polynomial features
            df['feature1_squared'] = df['feature1'] ** 2
            df['feature2_squared'] = df['feature2'] ** 2
            
            # 3. Scale numerical features
            numerical_features = ['feature1', 'feature2', 'interaction', 
                                'feature1_squared', 'feature2_squared']
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            
            logger.info("Features created successfully")
            return df
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise

    def save_features(self, df):
        """Save the processed features."""
        try:
            # Create processed data directory if it doesn't exist
            processed_dir = "data/processed"
            os.makedirs(processed_dir, exist_ok=True)
            
            # Save processed data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(processed_dir, f"processed_features_{timestamp}.csv")
            df.to_csv(output_path, index=False)
            
            logger.info(f"Processed features saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

def main():
    try:
        feature_builder = FeatureBuilder()
        
        # Load data
        df = feature_builder.load_data("/home/x-raf/Projects/mlops/mlops-pipeline/training-model-pipeline/data/raw")
        
        # Create features
        df_processed = feature_builder.create_features(df)
        
        # Save processed features
        feature_builder.save_features(df_processed)
        
        logger.info("Feature engineering pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 