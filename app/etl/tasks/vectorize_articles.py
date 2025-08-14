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
from typing import List
from uuid import uuid4
from app.singleton import ModelManager
from app.nlp import VectorDBManager
from app.etl_data.etl_models.feed_model import FeedModel
from app.config import get_service_logger


class VectorizeArticles:
    """ETL Task: Vectorize NewsArticle objects and store them using VectorDBManager."""

    def __init__(self, flashpoint_id: str):
        self.__flashpoint_id = flashpoint_id
        self.__embedding_model = ModelManager.get_embedding_model()
        self.__db = VectorDBManager()
        self._logger = get_service_logger("VectorizeArticles")

    # gettters
    def get_flashpoint_id(self) -> str:
        return self.__flashpoint_id

    def get_embedding_model(self):
        return self.__embedding_model

    def get_db(self) -> VectorDBManager:
        return self.__db

    def run(self, feeds: List[FeedModel]) -> str:
        """
        Vectorize and store NewsArticles in Chroma vector DB.

        Uses `.summary` if available, otherwise falls back to `.raw_text`.

        Returns:
            str: The Chroma collection name used
        """
        texts, metadatas, ids = [], [], []

        try:
            for feed in feeds:
                text = feed.summary.strip()
                if not text:
                    continue

                texts.append(text)
                metadatas.append(
                    {
                        "url": feed.url,
                        "domain": feed.domain,
                        "sourcecountry": feed.sourcecountry,
                        "language": feed.language or "unknown",
                    }
                )
                ids.append(str(uuid4()))  # each vector gets a unique ID why?

            if not texts:
                self._logger.info("No valid articles to vectorize.")
                return self.__flashpoint_id

            # Embed and insert into Chroma
            self._logger.info(
                f"Vectorizing {len(texts)} articles into collection: {self.__flashpoint_id}"
            )
            embeddings = self.__embedding_model.encode(
                texts, batch_size=32, show_progress_bar=True
            )

            self._logger.info(
                f"Inserting {len(texts)} articles into collection: {self.__flashpoint_id}"
            )
            self.__db.insert_documents(
                collection_name=self.__flashpoint_id,
                texts=texts,  # Stored as "documents"  use include=["documents"]
                metadatas=metadatas,  # Stored as "metadatas"  use include=["metadatas"]
                ids=ids,
                embeddings=embeddings,
            )

            self._logger.info(
                f"Vectorized {len(texts)} articles into collection: {self.__flashpoint_id}"
            )
            return self.__flashpoint_id

        except Exception as e:
            self._logger.error(f"Error vectorizing articles: {e}")
            return self.__flashpoint_id
