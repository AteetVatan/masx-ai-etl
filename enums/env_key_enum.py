"""Enum for the environment variables keys."""

from enum import Enum


class EnvKeyEnum(Enum):
    """Enum for the environment variables keys."""

    DEBUG_MODE = "DEBUG_MODE"
    PROXY_WEBPAGE = "PROXY_WEBPAGE"
    PROXY_TESTING_URL = "PROXY_TESTING_URL"
    MAX_WORKERS = "MAX_WORKERS"
    MASX_GDELT_API_KEY = "MASX_GDELT_API_KEY"
    MASX_GDELT_API_URL = "MASX_GDELT_API_URL"
    MASX_GDELT_API_KEYWORDS = "MASX_GDELT_API_KEYWORDS"
    MASX_GDELT_MAX_RECORDS = "MASX_GDELT_MAX_RECORDS"
