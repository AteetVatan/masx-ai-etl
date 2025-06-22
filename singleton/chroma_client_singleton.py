"""Singleton for Chroma client — supports local dev & Hugging Face prod modes."""
import os
import shutil
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from singleton import EnvManager
from enums import EnvKeyEnum
from typing import Optional


class ChromaClientSingleton:
    """Singleton for Chroma client — supports local dev & Hugging Face prod modes."""
    _instance: Optional[ChromaClient] = None

    @classmethod
    def get_client(cls) -> ChromaClient:
        """Get or initialize the Chroma client with appropriate persist path."""
        if cls._instance is None:
            persist_dir = cls.__get_persist_path()
            os.makedirs(persist_dir, exist_ok=True)

            cls._instance = ChromaClient(Settings(persist_directory=persist_dir))

        return cls._instance
    
    @classmethod
    def get_collection_if_exists(cls, collection_name: str):
        existing = cls.get_client().list_collections()
        collection_names = [col.name for col in existing]

        if collection_name in collection_names:
            return cls.get_client().get_or_create_collection(collection_name)
        else:
            raise ValueError(f"Chroma collection '{collection_name}' does not exist.")


    @classmethod
    def cleanup_chroma(cls):
        """Delete the Chroma persistence directory after ETL completion."""
        persist_path = cls.__get_persist_path()

        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
            print(f"Chroma vector DB deleted at: {persist_path}")
            

    @classmethod
    def __get_persist_path(cls) -> str:
        """Get the persist path for the Chroma client."""
        env = EnvManager.get_env_vars()
        is_debug = env.get(EnvKeyEnum.DEBUG_MODE.value, "false").lower() == "true"

        persist_path = (
            env.get(EnvKeyEnum.CHROMA_DEV_PERSIST_DIR.value, "./.chroma_storage")
            if is_debug
            else env.get(EnvKeyEnum.CHROMA_PROD_PERSIST_DIR.value, "/tmp/chroma")
        )
        return persist_path