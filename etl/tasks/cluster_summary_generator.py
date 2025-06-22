"""Cluster the vectorized documents stored in ChromaDB and generate a concise summary for each cluster."""

from collections import defaultdict, Counter
from singleton import ModelManager, ChromaClientSingleton
import numpy as np
from tqdm import tqdm
import torch
from nlp import BaseClusterer


class ClusterSummaryGenerator:
    """
    Generates cluster summaries from a ChromaDB collection,
    using pluggable clustering strategy (KMeans, HDBSCAN).
    
    to do
    Skip clusters with <2 docs
    -----Good for noise and performance.

    Parallelize Summarization (optional)
    Each cluster’s summary generation could be parallelized via concurrent.futures.


    """

    def __init__(self, collection_name: str, clustering_strategy: BaseClusterer):
        self.client = ChromaClientSingleton.get_client()
        self.collection = ChromaClientSingleton.get_collection_if_exists(collection_name)
        self.clusterer = clustering_strategy


    def generate(self) -> list[dict]:
        """
        The collection is a ChromaDB collection.
        The collection has the following fields:
        - documents: Raw documents (e.g., article summaries or full texts).
        - embeddings: Precomputed embeddings.
        - metadatas: Metadata (language, domain, URL, etc.).

        We need to cluster the documents using the clustering strategy and generate a summary for each cluster.
        """
        print("Fetching documents from Chroma...")
        docs = self.collection.get(include=["documents", "embeddings", "metadatas"])

        embeddings = np.array(docs["embeddings"])
        documents = docs["documents"]
        metadatas = docs["metadatas"]

        print(f"Clustering {len(documents)} articles...")
        cluster_labels = self._cluster_embeddings(embeddings)

        print("Generating cluster summaries...")
        return self._generate_cluster_summaries(documents, metadatas, cluster_labels)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        return self.clusterer.cluster(embeddings)

    def _generate_cluster_summaries(
        self,
        documents: list[str],
        metadatas: list[dict],
        labels: list[int]
    ) -> list[dict]:

        grouped_docs = defaultdict(list)
        grouped_meta = defaultdict(list)

        for doc, meta, label in zip(documents, metadatas, labels):
            if label == -1:
                continue  # Noise in HDBSCAN
            grouped_docs[label].append(doc)
            grouped_meta[label].append(meta)

        model, tokenizer, device = ModelManager.get_summarization_model()
        results = []

        for cluster_id in tqdm(grouped_docs, desc="Summarizing Clusters"):
            texts = grouped_docs[cluster_id]
            meta = grouped_meta[cluster_id]

            if len(texts) < 2:
                continue  # Skip trivial clusters

            joined = " ".join(texts)[:2048]  # Char-safe truncation
            input_ids = tokenizer.encode(joined, return_tensors="pt", truncation=True).to(device)
            summary_ids = model.generate(input_ids, max_length=150, min_length=30)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            top_domains = [d for d, _ in Counter(m.get("domain", "unknown") for m in meta).most_common(3)]
            langs = sorted(set(m.get("language", "unknown") for m in meta))
            urls = [m.get("url") for m in meta if m.get("url")]
            sample_urls = urls[:3]

            results.append({
                "cluster_id": cluster_id,
                "summary": summary,
                "article_count": len(texts),
                "top_domains": top_domains,
                "languages": langs,
                "sample_urls": sample_urls
            })

        return results
