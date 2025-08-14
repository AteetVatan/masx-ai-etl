# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                       │
# │  Project: MASX AI – Strategic Agentic AI System               │
# │  All rights reserved.                                         │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com
"""
This class is used to load the environment variables.
"""

import os
from dotenv import load_dotenv
from app.enums import EnvKeyEnum


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
            EnvKeyEnum.DEBUG.value: os.getenv(EnvKeyEnum.DEBUG.value),
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
