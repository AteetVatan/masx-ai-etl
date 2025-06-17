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
            EnvKeyEnum.DEBUG_MODE.value: os.getenv(EnvKeyEnum.DEBUG_MODE.value),
            EnvKeyEnum.PROXY_WEBPAGE.value: os.getenv(EnvKeyEnum.PROXY_WEBPAGE.value),
            EnvKeyEnum.PROXY_TESTING_URL.value: os.getenv(
                EnvKeyEnum.PROXY_TESTING_URL.value
            ),
            EnvKeyEnum.MAX_WORKERS.value: int(os.getenv(EnvKeyEnum.MAX_WORKERS.value)),
            EnvKeyEnum.MASX_GDELT_API_KEY.value: os.getenv(
                EnvKeyEnum.MASX_GDELT_API_KEY.value
            ),
            EnvKeyEnum.MASX_GDELT_API_URL.value: os.getenv(
                EnvKeyEnum.MASX_GDELT_API_URL.value
            ),
            EnvKeyEnum.MASX_GDELT_API_KEYWORDS.value: os.getenv(
                EnvKeyEnum.MASX_GDELT_API_KEYWORDS.value
            ),
            EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value: os.getenv(
                EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value, "10"
            ),
        }
        self.context.push(DagContextEnum.ENV_CONFIG.value, env_vars)
