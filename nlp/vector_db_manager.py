from typing import List, Optional
from uuid import uuid4
from singleton import ChromaClientSingleton


class VectorDBManager:
    """High-level interface for vector DB operations (currently Chroma)."""

    def __init__(self):
        self.client = ChromaClientSingleton.get_client()

    def insert_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):
        """Insert or upsert documents into a collection."""
        collection = self.client.get_or_create_collection(name=collection_name)

        # Generate UUIDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

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
        collection = self.client.get_or_create_collection(name=collection_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)

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
        self.client.delete_collection(name=collection_name)

    def list_collections(self) -> List[str]:
        """List all existing Chroma collections."""
        return [col.name for col in self.client.list_collections()]
