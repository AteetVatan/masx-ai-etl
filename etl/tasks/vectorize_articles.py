from typing import List
from uuid import uuid4
from singleton import ModelManager
from nlp import VectorDBManager
from schemas.news_article import NewsArticle  # adjust path if needed


class VectorizeArticles:
    """ETL Task: Vectorize NewsArticle objects and store them using VectorDBManager."""

    def __init__(self, collection_name: str = None):
        self.__collection_name = collection_name or f"masx_articles_{uuid4()}"
        self.__embedding_model = ModelManager.get_embedding_model()
        self.__db = VectorDBManager()


    #gettters
    def get_collection_name(self) -> str:
        return self.__collection_name
    
    def get_embedding_model(self):   
        return self.__embedding_model
    
    def get_db(self) -> VectorDBManager:
        return self.__db

    def run(self, articles: List[NewsArticle]) -> str:
        """
        Vectorize and store NewsArticles in Chroma vector DB.

        Uses `.summary` if available, otherwise falls back to `.raw_text`.

        Returns:
            str: The Chroma collection name used
        """
        texts, metadatas, ids = [], [], []

        for article in articles:
            text = article.summary.strip()
            if not text:
                continue

            texts.append(text)
            metadatas.append({
                "url": article.url,
                "domain": article.domain,
                "sourcecountry": article.sourcecountry,
                "language": article.language or "unknown"
            })
            ids.append(str(uuid4()))  # each vector gets a unique ID

        if not texts:
            print("No valid articles to vectorize.")
            return self.__collection_name

        # Embed and insert into Chroma
        embeddings = self.__embedding_model.encode(texts, batch_size=32, show_progress_bar=True)
        self.__db.insert_documents(
            collection_name=self.__collection_name,
            texts=texts, #Stored as "documents"  use include=["documents"]
            metadatas=metadatas, #Stored as "metadatas"  use include=["metadatas"]
            ids=ids,
            embeddings=embeddings
        )

        print(f"Vectorized {len(texts)} articles into collection: {self.__collection_name}")
        return self.__collection_name
