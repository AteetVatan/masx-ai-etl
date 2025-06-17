"""
MASX AI News ETL DAG

This DAG is used to extract news articles from the GDELT API,
extract the content of the articles, and summarize the articles.

The DAG is scheduled to run daily at 12:00 AM.
"""

# ─── Monkey Patch for Python 3.12 ────────────────────────────────────────────────
import os
import subprocess
from pathlib import Path
import logging
from datetime import datetime, timedelta

# ───  Python's logging module ────────────────────────────────────────────────────
logging.basicConfig(
    filename="masx_etl.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log(msg: str):
    print(msg)
    logging.info(msg)
    
# Monkey patch for Windows + Python 3.12 compatibility with Airflow
if not hasattr(os, "register_at_fork"):
    def dummy_register_at_fork(*args, **kwargs):
        print("os.register_at_fork not available on Python 3.12 — monkey-patched.")
        logging.warning("os.register_at_fork not available on Python 3.12 — monkey-patched.")
    os.register_at_fork = dummy_register_at_fork

# Check if Airflow DB is initialized (default SQLite check)
airflow_home = Path(os.environ.get("AIRFLOW_HOME", str(Path.home() / "airflow")))
db_file = airflow_home / "airflow.db"

if not db_file.exists():
    logging.info("Airflow DB not initialized. Running 'airflow db init'...")
    try:
        subprocess.run(["airflow", "db", "init"], check=True)
        logging.info("Airflow DB initialized successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to initialize Airflow DB: {e}")
else:
    logging.info("Airflow DB already initialized.")




# ─── Airflow - run Python functions in DAG tasks ──────────────────────────────────
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─── MASX AI ETL Modules - ETL components ───────────────────────────────────────
from etl import DAGContext, EnvManager, NewsManager, NewsContentExtractor, Summarizer

# ─── DAG Config ─────────────────────────────────────────────────────────────────
default_args = {
    "owner": "masx_ai",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "masx_news_etl_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)


# ─── Task Wrapper ───────────────────────────────────────────────────────────────
def init_context_and_run(func):
    def wrapper(**kwargs):
        log(f"Task {func.__name__} started.")
        context = DAGContext(kwargs)
        try:
            start = datetime.now()
            result = func(context, **kwargs)
            duration = (datetime.now() - start).total_seconds()
            log(f"Task {func.__name__} completed in {duration:.2f} seconds.")
            return result
        except Exception as e:
            logging.exception(f"Task {func.__name__} failed: {e}")
            print(f"Task `{func.__name__}` failed with error: {e}")
            raise

    return wrapper


# ─── Task Definitions ───────────────────────────────────────────────────────────
def run_env_manager(context: DAGContext, **kwargs):
    """
    Initialize the environment manager.
    """
    EnvManager(context)


env_manager_task = PythonOperator(
    task_id="env_manager",
    python_callable=init_context_and_run(run_env_manager),
    dag=dag,
)


def run_extract_news(context: DAGContext, **kwargs):
    """
    Extract news articles from the GDELT API.
    """
    news_manager = NewsManager(context)
    news_manager.news_articles()


# ─── Task Definitions ───────────────────────────────────────────────────────────
extract_news_task = PythonOperator(
    task_id="extract_news",
    python_callable=init_context_and_run(run_extract_news),
    dag=dag,
)


def run_extract_articles(context: DAGContext, **kwargs):
    """
    Extract the content of the news articles.
    """
    article_extractor = NewsContentExtractor(context)
    article_extractor.extract_articles()


extract_articles_task = PythonOperator(
    task_id="extract_articles",
    python_callable=init_context_and_run(run_extract_articles),
    dag=dag,
)


def run_summarize_articles(context: DAGContext, **kwargs):
    """
    Summarize the news articles.
    """
    summarizer = Summarizer(context)
    summarizer.summarize_all_articles()


summarize_articles_task = PythonOperator(
    task_id="summarize_articles",
    python_callable=init_context_and_run(run_summarize_articles),
    dag=dag,
)


# ─── DAG Task Flow ──────────────────────────────────────────────────────────────
env_manager_task >> extract_news_task >> extract_articles_task >> summarize_articles_task
