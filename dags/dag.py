import datetime
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.sdk import Variable
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline_dvc_minio',
    default_args=default_args,
    description='ML pipeline with DVC and MinIO',
    schedule=timedelta(days=1),
    catchup=False,
)

def init_dvc(**context):
    import subprocess
    repo_path = Variable.get("REPO_PATH", default="/opt/airflow")
    os.chdir(repo_path)

    try:
        if not os.path.exists('.git'):
            result = subprocess.run(['git', 'init'], capture_output=True, text=True, check=True)
            print("Git initialized.")
            subprocess.run(['git', 'config', 'init.defaultBranch', 'main'], check=True)

        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            print("DVC initialized.")

        remote_name = "dmitrij-novikov-glf7779"
        bucket = Variable.get("MINIO_BUCKET", default="dmitrij-novikov-glf7779")
        s3_url = f"s3://{bucket}"
        access_key = Variable.get("MINIO_ACCESS_KEY")
        secret_key = Variable.get("MINIO_SECRET_KEY")
        endpoint = Variable.get("MINIO_ENDPOINT", default="https://s3.lab.karpov.courses")

        subprocess.run(['dvc', 'remote', 'add', '-f', '-d', remote_name, s3_url], check=True)
        print(f"DVC remote '{remote_name}' added/overwritten with URL {s3_url}.")

        subprocess.run([
            'dvc', 'config', f'remote.{remote_name}.access_key_id', access_key
        ], check=True)
        subprocess.run([
            'dvc', 'config', f'remote.{remote_name}.secret_access_key', secret_key
        ], check=True)
        subprocess.run([
            'dvc', 'config', f'remote.{remote_name}.endpointurl', endpoint
        ], check=True)

        print(f"DVC remote '{remote_name}' configured successfully with endpoint {endpoint}.")

        if os.path.exists('data.dvc'):
            subprocess.run(['dvc', 'pull'], check=True)
            print("Data pulled from remote.")

    except subprocess.CalledProcessError as e:
        print(f"DVC/Git error: {e.stderr}")
        raise

def load_data(**kwargs):
    import subprocess
    import pandas as pd
    repo_path = Variable.get("REPO_PATH", default="/opt/airflow")
    os.chdir(repo_path)

    dvc_file = 'data/winequality-red.csv.dvc'
    if not os.path.exists(dvc_file):
        logging.warning(f"{dvc_file} not found. Skipping pull.")
        return None

    try:
        logging.info(f"Pulling data from DVC... (file: {dvc_file})")
        subprocess.run(['dvc', 'pull'], check=True)
        df = pd.read_csv('data/winequality-red.csv')
        logging.info(f"Data loaded: {len(df)} rows.")
        return df
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC pull error: {e.stderr}")
        raise


def train_model(**kwargs):
    import subprocess
    import json
    from datetime import datetime
    import logging
    import os
    
    repo_path = Variable.get('REPO_PATH', default='/opt/airflow')
    os.makedirs(repo_path, exist_ok=True)
    os.chdir(repo_path)
    
    env = os.environ.copy()
    env['MODEL_PATH'] = 'models/model.pkl'
    
    try:
        subprocess.run(['python', 'src/train.py'], check=True, env=env)
        logging.info("Model trained successfully via src/train.py.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running src/train.py: {e}")
        raise
    
    metadata = {
        'timestamp': str(datetime.now()),
        'version': '1.0',
        'description': 'Trained ML model'
    }
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    lock_path = '.dvc/lock'
    if os.path.exists(lock_path):
        os.remove(lock_path)
        logging.info("DVC lock cleared.")
    
    try:
        subprocess.run(['dvc', 'add', 'models/model.pkl'], check=True)
        subprocess.run(['dvc', 'add', 'models/metrics.json'], check=True)
        subprocess.run(['dvc', 'add', 'metadata.json'], check=True)
        subprocess.run(['dvc', 'push'], check=True)
        logging.info("Model, metrics, and metadata pushed to DVC remote.")
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC error: {e}")
        raise


init_dvc_task = PythonOperator(
    task_id='init_dvc',
    python_callable=init_dvc,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

init_dvc_task >> load_data_task >> train_model_task
