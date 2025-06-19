"""
Module to load the environment variables.
"""

import os
from dotenv import load_dotenv
from enums import EnvKeyEnum, EnvModeEnum


class EnvManager:
    """
    Class to load and access environment variables.
    """
    _env_vars: dict[EnvKeyEnum, str] = {}

    @classmethod
    def get_env_vars(cls) -> dict[str, str]:
        """
        Lazily load and return environment variables as a dictionary.
        """
        if not cls._env_vars:
            cls.__load_env()
        return cls._env_vars

    @classmethod
    def __load_env(cls):
        """
        Load the environment variables from .env into the class dictionary.
        """
        load_dotenv()

        try:
            cls._env_vars = {
                EnvKeyEnum.ENV_MODE.value: str(
                    EnvModeEnum(os.getenv(EnvKeyEnum.ENV_MODE.value, EnvModeEnum.DEV.value))
                ),
                EnvKeyEnum.PROXY_WEBPAGE.value: os.getenv(EnvKeyEnum.PROXY_WEBPAGE.value, ""),
                EnvKeyEnum.PROXY_TESTING_URL.value: os.getenv(EnvKeyEnum.PROXY_TESTING_URL.value, ""),
                EnvKeyEnum.MAX_WORKERS.value: str(int(os.getenv(EnvKeyEnum.MAX_WORKERS.value, "20"))),
                EnvKeyEnum.MASX_GDELT_API_KEY.value: os.getenv(EnvKeyEnum.MASX_GDELT_API_KEY.value, ""),
                EnvKeyEnum.MASX_GDELT_API_URL.value: os.getenv(EnvKeyEnum.MASX_GDELT_API_URL.value, ""),
                EnvKeyEnum.MASX_GDELT_API_KEYWORDS.value: os.getenv(EnvKeyEnum.MASX_GDELT_API_KEYWORDS.value, ""),
                EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value: str(
                    int(os.getenv(EnvKeyEnum.MASX_GDELT_MAX_RECORDS.value, "10"))
                ),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load environment variables: {e}")

