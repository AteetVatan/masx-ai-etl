# news_etl_dag_debug.py — Local Debug Runner for MASX News ETL
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Monkey patch for Python 3.12 compatibility with Airflow
if not hasattr(os, "register_at_fork"):

    def dummy_register_at_fork(*args, **kwargs):
        print("os.register_at_fork is not available on Python 3.12 — monkey-patched.")

    os.register_at_fork = dummy_register_at_fork


from enums import DagContextEnum
from etl import (
    DAGContext,
    EnvManager,
    NewsManager,
    NewsContentExtractor,
    Summarizer,
)


class FakeTaskInstance:
    """
    Simulates Airflow's TaskInstance for local debugging.
    """

    def __init__(self):
        self.memory_store = {}

    def xcom_push(self, key, value):
        print(f"[XCOM PUSH] key='{key}' | type={type(value)}")
        self.memory_store[key] = value

    def xcom_pull(self, key=None, task_ids=None):
        print(f"[XCOM PULL] key='{key}' | from task_id='{task_ids}'")
        return self.memory_store.get(key)


def run_debug():
    print("🔧 Starting MASX News ETL — Local Debug Mode")

    fake_kwargs = {"ti": FakeTaskInstance()}
    context = DAGContext(fake_kwargs)

    try:
        # get the time now
        start_time = time.time()
        # Mock data
        from mock_data.mock_data import mock_data

        context.push(DagContextEnum.NEWS_ARTICLES.value, mock_data)

        print("\n Running EnvManager...")
        EnvManager(context)
        # self.context.push(DagContextEnum.ENV_CONFIG.value, env_vars)
        env_vars = context.pull(DagContextEnum.ENV_CONFIG.value)

        print("\n Running NewsManager...")
        news_mgr = NewsManager(context)
        news_mgr.news_articles()
        # self.context.push(DagContextEnum.NEWS_ARTICLES.value, news_articles.dict())
        news_articles = context.pull(DagContextEnum.NEWS_ARTICLES.value)

        print("\n Running NewsContentExtractor...")
        extractor = NewsContentExtractor(context)
        extractor.extract_articles()
        # self.context.push(DagContextEnum.NEWS_ARTICLE_WITH_DESC.value, scraped_articles)
        scraped_articles = context.pull(DagContextEnum.NEWS_ARTICLE_WITH_DESC.value)

        print("\n Running Summarizer...")
        summarizer = Summarizer(context)
        summarizer.summarize_all_articles()
        # self.context.push(DagContextEnum.NEWS_ARTICLE_WITH_SUMMARY.value, serialized)
        summarized = context.pull(DagContextEnum.NEWS_ARTICLE_WITH_SUMMARY.value)

        # get the time now
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

    except Exception as e:
        print(f"\n Debug run failed with error:\n{e}")
    else:
        print("\n Debug run completed successfully.")


if __name__ == "__main__":
    run_debug()
