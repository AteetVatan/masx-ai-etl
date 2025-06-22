"""
This class is used to load the environment variables.
"""

import os
from dotenv import load_dotenv
from enums import EnvKeyEnum


class EnvManager:
    """
    This class is used to load the environment variables.
    """

    __env_vars = {}

    @classmethod
    def get_env_vars(cls):
        """
        Get the environment variables.
        """
        if not cls.__env_vars:
            cls.load_env_vars()
        return cls.__env_vars

    @classmethod
    def load_env_vars(cls):
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
            EnvKeyEnum.CHROMA_DEV_PERSIST_DIR.value: os.getenv(
                EnvKeyEnum.CHROMA_DEV_PERSIST_DIR.value
            ),
            EnvKeyEnum.CHROMA_PROD_PERSIST_DIR.value: os.getenv(
                EnvKeyEnum.CHROMA_PROD_PERSIST_DIR.value
            ),
        }
        cls.__env_vars = env_vars
