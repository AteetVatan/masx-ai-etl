"""
This class is used to load the environment variables.
"""

import os
from dotenv import load_dotenv
from etl.dag_context import DAGContext
from enums import EnvKeyEnum, DagContextEnum


class EnvManager:
    """
    This class is used to load the environment variables.
    """

    def __init__(self, context: DAGContext):
        self.context = context
        self.load_env()

    def load_env(self):
        """
        Load the environment variables.
        """
        load_dotenv()
        env_vars = {
            EnvKeyEnum.PROXY_WEBPAGE.value: os.getenv("PROXY_WEBPAGE"),
            EnvKeyEnum.PROXY_TESTING_URL.value: os.getenv("PROXY_TESTING_URL"),
            EnvKeyEnum.MAX_WORKERS.value: int(os.getenv("MAX_WORKERS", 10)),
            EnvKeyEnum.MASX_GDELT_API_KEY.value: os.getenv("MASX_GDELT_API_KEY"),
            EnvKeyEnum.MASX_GDELT_API_URL.value: os.getenv("MASX_GDELT_API_URL"),
            EnvKeyEnum.MASX_GDELT_KEYWORDS.value: os.getenv(
                "MASX_GDELT_KEYWORDS", "Germany"
            ),
            EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value: os.getenv(
                "MASX_GDELT_MAX_RECORDS", "10"
            ),
        }
        self.context.push(DagContextEnum.ENV_CONFIG.value, env_vars)
