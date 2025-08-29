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

"""This module provides a high-level interface for vector DB operations (currently Chroma)."""

from uuid import uuid4
from typing import List, Optional

from app.config import get_service_logger
from app.singleton import ChromaClientSingleton


class VectorDBManager:
    """High-level interface for vector DB operations (currently Chroma)."""

    def __init__(self):
        self.client = ChromaClientSingleton.get_client()
        self._logger = get_service_logger("VectorDBManager")

    def insert_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Insert or upsert documents into a collection."""
        try:
            self._logger.info(
                f"vector_db_manager.py:Inserting documents into collection: {collection_name}"
            )
            collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            self._logger.error(
                f"vector_db_manager.py:Error getting or creating collection: {e}"
            )
            raise e

        # Generate UUIDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        self._logger.info(
            f"vector_db_manager.py:Adding documents to collection: {collection_name}"
        )
        collection.add(
            documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings
        )

    def query_similar(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[dict]:
        """Query similar documents from a collection."""
        try:
            self._logger.info(
                f"vector_db_manager.py:Querying collection: {collection_name}"
            )
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as e:
            self._logger.error(f"vector_db_manager.py:Error querying collection: {e}")
            raise e

        return [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]

    def delete_collection(self, collection_name: str):
        """Delete a collection entirely."""
        try:
            self._logger.info(
                f"vector_db_manager.py:Deleting collection: {collection_name}"
            )
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            self._logger.error(f"vector_db_manager.py:Error deleting collection: {e}")
            raise e

    def list_collections(self) -> List[str]:
        """List all existing Chroma collections."""
        try:
            self._logger.info("vector_db_manager.py:Listing collections")
            return [col.name for col in self.client.list_collections()]
        except Exception as e:
            self._logger.error(f"vector_db_manager.py:Error listing collections: {e}")
            raise e
