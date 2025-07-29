from typing import List, Optional
from uuid import uuid4
from singleton import ChromaClientSingleton
from config import get_service_logger


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
            self._logger.info(f"Inserting documents into collection: {collection_name}")
            collection = self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            self._logger.error(f"Error getting or creating collection: {e}")
            raise e

        # Generate UUIDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        self._logger.info(f"Adding documents to collection: {collection_name}")
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def query_similar(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5
    ) -> List[dict]:
        """Query similar documents from a collection."""
        try:
            self._logger.info(f"Querying collection: {collection_name}")
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as e:
            self._logger.error(f"Error querying collection: {e}")
            raise e

        return [
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta
            }
            for doc_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ]

    def delete_collection(self, collection_name: str):
        """Delete a collection entirely."""
        try:
            self._logger.info(f"Deleting collection: {collection_name}")
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            self._logger.error(f"Error deleting collection: {e}")
            raise e

    def list_collections(self) -> List[str]:
        """List all existing Chroma collections."""
        try:
            self._logger.info("Listing collections")
            return [col.name for col in self.client.list_collections()]
        except Exception as e:
            self._logger.error(f"Error listing collections: {e}")
            raise e
       
