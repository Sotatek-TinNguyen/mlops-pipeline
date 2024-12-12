import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data',
        'data/raw',
        'data/processed'
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        else:
            logger.info(f"Directory already exists: {dir_path}")

def generate_sample_data(n_samples=1000):
    """Generate sample data for a binary classification problem."""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(35, 10, n_samples).round(2)
    income = np.random.normal(50000, 20000, n_samples).round(2)
    credit_score = np.random.normal(700, 100, n_samples).round(2)
    
    # Generate target (loan approval: 0 = rejected, 1 = approved)
    probability = 1 / (1 + np.exp(-(
        -5 +
        0.1 * (age - 35) +
        0.3 * ((income - 50000) / 10000) +
        0.2 * ((credit_score - 700) / 100)
    )))
    target = (np.random.random(n_samples) < probability).astype(int)
    
    # Create DataFrame
    data = {
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'loan_approved': target
    }
    
    return pd.DataFrame(data)

def save_data(df, data_type='raw'):
    """Save data to CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loan_data_{timestamp}.csv"
    
    if data_type == 'raw':
        filepath = os.path.join('data', 'raw', filename)
    else:
        filepath = os.path.join('data', 'processed', filename)
    
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved to {filepath}")
    return filepath

def main():
    try:
        # Create directories
        create_directories()
        
        # Generate and save raw data
        logger.info("Generating sample data...")
        df = generate_sample_data()
        
        # Save raw data
        filepath = save_data(df, 'raw')
        
        # Display sample of the data
        logger.info("\nSample of generated data:")
        print(df.head())
        
        # Display basic statistics
        logger.info("\nBasic statistics:")
        print(df.describe())
        
        logger.info("\nData generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 