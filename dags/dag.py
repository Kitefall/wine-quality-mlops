import datetime
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import subprocess
import pandas as pd

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

def init_git(**kwargs):
    from airflow.sdk import Variable
    repo_path = Variable.get('REPO_PATH', '/opt/airflow')
    os.makedirs(repo_path, exist_ok=True)
    os.chdir(repo_path)
    
    try:
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            logging.info("Git repository initialized.")
        else:
            logging.info("Git repository already exists.")
        
        subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', repo_path], check=True)
        logging.info("Git safe.directory configured.")
        
        remote_url = Variable.get('GIT_REMOTE_URL')
        if not remote_url:
            logging.error("GIT_REMOTE_URL variable not set.")
            raise ValueError("GIT_REMOTE_URL is required.")
        
        result = subprocess.run(['git', 'remote', 'set-url', 'origin', remote_url], check=True)
        if result.returncode != 0:
            subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True)
            logging.info("Git remote 'origin' added.")
        else:
            logging.info("Git remote 'origin' already exists.")
        
        git_user_name = Variable.get('GIT_USER_NAME')
        git_user_email = Variable.get('GIT_USER_EMAIL')
        if git_user_name:
            subprocess.run(['git', 'config', 'user.name', git_user_name], check=True)
        if git_user_email:
            subprocess.run(['git', 'config', 'user.email', git_user_email], check=True)
        logging.info("Git user configured.")
        
        branch = Variable.get('GIT_BRANCH', 'develop')
        subprocess.run(['git', 'pull', 'origin', branch], check=True)
        logging.info("Git repository synced with remote.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git init/remote/pull error: {e}")
    except ValueError as e:
        logging.error(str(e))
        raise

def init_dvc(**context):
    from airflow.sdk import Variable
    repo_path = Variable.get("REPO_PATH", "/opt/airflow")
    os.chdir(repo_path)

    try:
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], capture_output=True, text=True, check=True)
            subprocess.run(['git', 'config', 'init.defaultBranch', 'main'], check=True)
            logging.info("Git initialized.")

        if not os.path.exists('.dvc'):
            subprocess.run(['dvc', 'init'], check=True)
            logging.info("DVC initialized.")

        remote_name = Variable.get("REMOTE_NAME", "dmitrij-novikov-glf7779")
        bucket = Variable.get("MINIO_BUCKET", "dmitrij-novikov-glf7779")
        s3_url = f"s3://{bucket}"
        access_key = Variable.get("MINIO_ACCESS_KEY")
        secret_key = Variable.get("MINIO_SECRET_KEY")
        endpoint = Variable.get("MINIO_ENDPOINT", "https://s3.lab.karpov.courses")

        if not access_key or not secret_key:
            raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set.")

        subprocess.run(['dvc', 'remote', 'add', '-f', '-d', remote_name, s3_url], check=True)
        logging.info(f"DVC remote '{remote_name}' added/overwritten with URL {s3_url}.")

        subprocess.run(['dvc', 'config', f'remote.{remote_name}.access_key_id', access_key], check=True)
        subprocess.run(['dvc', 'config', f'remote.{remote_name}.secret_access_key', secret_key], check=True)
        subprocess.run(['dvc', 'config', f'remote.{remote_name}.endpointurl', endpoint], check=True)
        logging.info(f"DVC remote '{remote_name}' configured successfully with endpoint {endpoint}.")

        try:
            result = subprocess.run(['dvc', 'pull', '--force'], capture_output=True, text=True, check=True)
            logging.info("DVC pull successful: " + result.stdout)
        except subprocess.CalledProcessError as e:
            logging.warning(f"DVC pull failed (may be first run): {e.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"DVC/Git error: {e.stderr}")
        raise
    except ValueError as e:
        logging.error(str(e))
        raise

def load_data(**kwargs):
    from airflow.sdk import Variable
    repo_path = Variable.get("REPO_PATH", "/opt/airflow")
    os.chdir(repo_path)

    data_path = 'data/winequality-red.csv'
    dvc_path = data_path + '.dvc'

    try:
        logging.info("Pulling data from DVC with --force...")
        result = subprocess.run(['dvc', 'pull', '--force'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"DVC pull failed: stdout={result.stdout}, stderr={result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, ['dvc', 'pull', '--force'], result.stdout, result.stderr)

        if not os.path.exists(data_path):
            logging.error(f"Data file {data_path} not found after pull.")
            raise FileNotFoundError(f"{data_path} missing.")

        df = pd.read_csv(data_path)
        logging.info(f"Data loaded: {len(df)} rows, shape {df.shape}.")
        return df

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Load data error: {str(e)}")
        raise

def train_model(**kwargs):
    from airflow.sdk import Variable
    import json
    from datetime import datetime
    import os
    
    repo_path = Variable.get('REPO_PATH', '/opt/airflow')
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
        'description': 'Trained ML model'
    }
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    lock_path = '.dvc/lock'
    if os.path.exists(lock_path):
        os.remove(lock_path)
        logging.info("DVC lock cleared.")
    
    try:
        subprocess.run(['dvc', 'add', 'models/model.pkl', 'models/metrics.json'], check=True)
        subprocess.run(['dvc', 'push'], check=True)
        logging.info("Model added and pushed to DVC.")
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC add/push error: {e}")
        raise
    
    try:
        subprocess.run(['git', 'add', 'models/model.pkl.dvc', 'models/metrics.json.dvc'], check=True)
        logging.info("Files added to Git staging area.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git add error: {e}")
        raise
    
    try:
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'], check=False)
        if result.returncode == 0:
            logging.info("No changes to commit in Git. Skipping commit.")
            return
        else:
            subprocess.run(['git', 'commit', '-m', 'Updated model and metrics'], check=True)
            logging.info("Git commit completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git commit error: {e}")
        raise
    
    try:
        result = subprocess.run(['git', 'push', 'origin', 'develop'], check=True, capture_output=True, text=True)
        if result.stdout:
            logging.info(result.stdout)
        if result.stderr:
            logging.info(result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error(f"Git push error: {e}")
        raise

init_dvc_task = PythonOperator(
    task_id='init_dvc',
    python_callable=init_dvc,
    dag=dag,
)

init_git_task = PythonOperator(
    task_id='init_git',
    python_callable=init_git,
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

init_git_task >> init_dvc_task >> load_data_task >> train_model_task
