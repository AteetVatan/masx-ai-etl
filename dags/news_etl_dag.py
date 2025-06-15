"""
This DAG is used to extract news articles from the GDELT API,
extract the content of the articles, and summarize the articles.

The DAG is scheduled to run daily at 12:00 AM.

The DAG is defined in the airflow.cfg file.

The DAG is run using the airflow command line interface.
"""

# Monkey patch os to prevent Airflow crash on Python 3.12
import os

if not hasattr(os, "register_at_fork"):

    def dummy_register_at_fork(*args, **kwargs):
        print("os.register_at_fork is not available on Python 3.12 — monkey-patched.")

    os.register_at_fork = dummy_register_at_fork

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from etl import DAGContext, EnvManager, NewsManager, NewsContentExtractor, Summarizer


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


def init_context_and_run(func):
    """
    Wrapper function to initialize the context and run the function.
    """

    def wrapper(**kwargs):
        context = DAGContext(kwargs)
        return func(context, **kwargs)

    return wrapper


def run_env_manager(context: DAGContext, **kwargs):
    """Actual task runner for env manager"""
    EnvManager(context)


# Airflow Task Definition
env_manager_task = PythonOperator(
    task_id="env_manager",
    python_callable=init_context_and_run(run_env_manager),
    dag=dag,
)
# self.context.push(DagContextEnum.ENV_CONFIG.value, env_vars)


def run_extract_news(context: DAGContext, **kwargs):
    """
    Actual Airflow task runner for NewsManager.
    Fetches, validates, and stores articles into DAGContext.
    """
    news_manager = NewsManager(context)
    news_manager.news_articles()


# NewsManager
extract_news_task = PythonOperator(
    task_id="extract_news",
    python_callable=init_context_and_run(run_extract_news),
    dag=dag,
)
# self.context.push(DagContextEnum.NEWS_ARTICLES.value, news_articles.dict())


# NewsContentExtractor
def run_extract_articles(context: DAGContext, **kwargs):
    """
    Actual Airflow task runner for NewsContentExtractor.
    Extracts raw text for each article using proxy-enabled scraping.
    """
    article_extractor = NewsContentExtractor(context)
    article_extractor.extract_articles()


extract_articles_task = PythonOperator(
    task_id="extract_articles",
    python_callable=init_context_and_run(run_extract_articles),
    dag=dag,
)
# self.context.push(DagContextEnum.NEWS_ARTICLE_WITH_DESC.value, scraped_articles)


# Summarizer
def run_summarize_articles(context: DAGContext, **kwargs):
    """
    Actual Airflow task runner for Summarizer.
    Summarizes the articles using the BART model.
    """
    summarizer = Summarizer(context)
    summarizer.summarize_all_articles()


summarize_articles_task = PythonOperator(
    task_id="summarize_articles",
    python_callable=init_context_and_run(run_summarize_articles),
    dag=dag,
)
# self.context.push(DagContextEnum.NEWS_ARTICLE_WITH_SUMMARY.value, serialized)


# Task Dependencies
(
    env_manager_task
    >> extract_news_task
    >> extract_articles_task
    >> summarize_articles_task
)
