from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='A model retraining pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Task 1: Data Collection and Validation
data_collection = BashOperator(
    task_id='data_collection',
    bash_command='python /opt/airflow/src/data/collect_data.py',
    dag=dag
)

# Task 2: Feature Engineering
feature_engineering = BashOperator(
    task_id='feature_engineering',
    bash_command='python /opt/airflow/src/features/build_features.py',
    dag=dag
)

# Task 3: Model Training
model_training = BashOperator(
    task_id='model_training',
    bash_command='python /opt/airflow/src/training/train_model.py',
    dag=dag
)

# Task 4: Model Evaluation
model_evaluation = BashOperator(
    task_id='model_evaluation',
    bash_command='python /opt/airflow/src/evaluation/evaluate_model.py',
    dag=dag
)

# Define task dependencies
data_collection >> feature_engineering >> model_training >> model_evaluation 