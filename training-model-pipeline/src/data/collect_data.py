import pandas as pd
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    data_dir = "data"
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    for dir_path in [data_dir, raw_dir, processed_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def collect_data():
    """Collect data from source and save to raw data directory."""
    try:
        # Here you would typically:
        # 1. Connect to your data source (database, API, etc.)
        # 2. Extract the data
        # 3. Perform initial validation
        # 4. Save the raw data
        
        # Example with dummy data
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 1]
        }
        df = pd.DataFrame(data)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/raw/data_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Data collected and saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        raise

def validate_data(file_path):
    """Perform data validation checks."""
    try:
        df = pd.read_csv(file_path)
        
        # Example validation checks
        assert not df.empty, "Data is empty"
        assert df.isnull().sum().sum() == 0, "Data contains null values"
        
        logger.info("Data validation passed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    setup_directories()
    raw_data_path = collect_data()
    if validate_data(raw_data_path):
        logger.info("Data collection and validation completed successfully")
    else:
        logger.error("Data collection and validation failed") 