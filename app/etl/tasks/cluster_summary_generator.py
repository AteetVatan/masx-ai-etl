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
"""Cluster the vectorized documents stored in ChromaDB and generate a concise summary for each cluster."""

from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from app.nlp import BaseClusterer
from app.singleton import ChromaClientSingleton
from typing import Tuple, List
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums
from app.core.concurrency import InferenceRuntime
from app.core.concurrency import RuntimeConfig
from app.core.model import SummarizationModelManager
from app.etl_data.etl_models import ClusterModel
from app.config import get_service_logger
from app.config import get_settings


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

    def __init__(self, flashpoint_id: str, clustering_strategy: BaseClusterer):
        self.settings = get_settings()
        self.client = ChromaClientSingleton.get_client()
        self.collection = ChromaClientSingleton.get_collection_if_exists(flashpoint_id)
        self.clusterer = clustering_strategy

        # Initialize logger

        self.logger = get_service_logger("ClusterSummaryGenerator")
        self.inference_runtime: InferenceRuntime = None
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)
        self.flashpoint_id = flashpoint_id

    async def generate(self) -> list[ClusterModel]:
        """
        The collection is a ChromaDB collection.
        The collection has the following fields:
        - documents: Raw documents (e.g., article summaries or full texts).
        - embeddings: Precomputed embeddings.
        - metadatas: Metadata (language, domain, URL, etc.).

        We need to cluster the documents using the clustering strategy and generate a summary for each cluster.
        """
        try:
            print("Fetching documents from Chroma...")
            docs = self.collection.get(include=["documents", "embeddings", "metadatas"])

            embeddings = np.array(docs["embeddings"])
            documents = docs["documents"]
            metadatas = docs["metadatas"]

            print(f"Clustering {len(documents)} articles...")
            cluster_labels = await self._cluster_embeddings(embeddings)

            # now generate cluster summaries

            print("Generating cluster summaries...")
            result: list[ClusterModel] = await self._generate_cluster_summaries(
                self.flashpoint_id, documents, metadatas, cluster_labels
            )

            return result

        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error generating cluster summaries: {e}"
            )
            raise e

    async def _cluster_embeddings(self, embeddings: np.ndarray) -> list[int]:
        try:
            # Use async cluster method to avoid asyncio.run() conflicts
            if hasattr(self.clusterer, "cluster_async"):
                # HDBSCAN has async method
                return await self.clusterer.cluster_async(embeddings)
            else:
                # KMeans only has sync method
                return self.clusterer.cluster(embeddings)
        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error clustering embeddings: {e}"
            )
            raise e

    async def _generate_cluster_summaries(
        self,
        flashpoint_id: str,
        documents: list[str],
        metadatas: list[dict],
        labels: list[int],
    ) -> list[ClusterModel]:
        try:

            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self.logger.info("Summarizer: initializing inference runtime")
                await self._initialize_inference_runtime()

            summarizer: SummarizationModelManager = self.inference_runtime.model_manager
            batch_size = summarizer.pool_size

            # Group docs by cluster label
            grouped_docs = defaultdict(list)
            grouped_meta = defaultdict(list)

            for doc, meta, label in zip(documents, metadatas, labels):
                if label == -1:
                    continue  # Noise in HDBSCAN
                grouped_docs[label].append(doc)
                grouped_meta[label].append(meta)

            cluster_ids = list(grouped_docs.keys())
            results: List[ClusterModel] = []

            # Process clusters in batches
            for i in range(0, len(cluster_ids), batch_size):
                batch_ids = cluster_ids[i : i + batch_size]
                self.logger.info(
                    f"Summarizer: processing batch of {len(batch_ids)} clusters"
                )

                # Gather cluster data for this batch
                batch_clusters = []
                for cluster_id in batch_ids:
                    texts = grouped_docs[cluster_id]
                    meta = grouped_meta[cluster_id]
                    if len(texts) < 2:
                        continue  # Skip trivial clusters
                    batch_clusters.append((cluster_id, texts, meta))

            results = []
            results.extend(await self._process_batch(flashpoint_id, batch_clusters))

            return results

        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error generating cluster summaries: {e}"
            )
            raise e

    async def _process_batch(
        self,
        flashpoint_id: str,
        batch_clusters: list[tuple[int, list[str], list[dict]]],
    ) -> list[ClusterModel]:
        try:
            # (cluster_id, texts, meta)
            summarizer: SummarizationModelManager = self.inference_runtime.model_manager
            # cluster in batches of model pool
            batch_size = summarizer.pool_size

            async def run(
                flashpoint_id: str, cluster_id: int, texts: list[str], meta: list[dict]
            ) -> ClusterModel | None:
                # each task acquires its own instance
                async with summarizer.acquire(destroy_after_use=False) as instance:
                    try:
                        result = await self.cpu_executors.run_in_thread(
                            self._generate_group_summaries,
                            flashpoint_id,
                            cluster_id,
                            texts,
                            meta,
                            instance.model,
                            instance.tokenizer,
                            instance.device,
                        )
                        if result is None:
                            raise Exception("Summarizer: Error summarizing cluster")

                        #         params = (
                        #     str(flashpoint_id),
                        #     int(cluster["cluster_id"]),
                        #     cluster["summary"],
                        #     int(cluster["article_count"]),
                        #     json.dumps(cluster.get("top_domains", [])),
                        #     json.dumps(cluster.get("languages", [])),
                        #     json.dumps(cluster.get("urls", [])),
                        #     json.dumps(cluster.get("images", [])),
                        # )

                        return result
                    except Exception as e:
                        self.logger.error(f"Summarizer: Error summarizing cluster: {e}")
                        return None

            tasks = [
                run(flashpoint_id, cluster_id, texts, meta)
                for cluster_id, texts, meta in batch_clusters
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error generating cluster summaries: {e}"
            )
            raise e

    def _generate_group_summaries(
        self, flashpoint_id, cluster_id, texts, meta, model, tokenizer, device
    ) -> list[dict]:
        try:

            # summary = summarize_cluster_dynamic(model, tokenizer, device, texts)

            summary = self.summarize_cluster_dynamic(
                model, tokenizer, device, texts, meta
            )

            # ********************

            # joined = " ".join(texts)[:2048]  # Char-safe truncation
            # input_ids = tokenizer.encode(
            #     joined, return_tensors="pt", truncation=True
            # ).to(device)
            # summary_ids = model.generate(input_ids, max_length=150, min_length=30)
            # summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # ********************

            top_domains = [
                d
                for d, _ in Counter(
                    m.get("domain", "unknown") for m in meta
                ).most_common(3)
            ]
            all_domains = sorted(set(m.get("domain", "unknown") for m in meta))

            langs = sorted(set(m.get("language", "unknown") for m in meta))
            urls = [m.get("url") for m in meta if m.get("url")]
            images = [
                m.get("image")
                for m in meta
                if m.get("image") and m.get("image") != "unknown"
            ]

            cluster_model = ClusterModel(
                flashpoint_id=flashpoint_id,
                cluster_id=cluster_id,
                summary=summary,
                article_count=len(texts),
                top_domains=top_domains,
                all_domains=all_domains,
                languages=langs,
                urls=urls,
                images=images,
            )

            # result = {
            #     "cluster_id": cluster_id,
            #     "summary": summary,
            #     "article_count": len(texts),
            #     "top_domains": top_domains,
            #     "all_domains": all_domains,
            #     "languages": langs,
            #     "urls": urls,
            #     "images": images,
            # }
            return cluster_model
        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error generating cluster summaries: {e}"
            )
            raise e

    def summarize_cluster_dynamic(self, model, tokenizer, device, texts, metas) -> str:
        try:
            src_cap, dec_cap = self._safe_caps(model, tokenizer)
            sep = "\n\n— DOC —\n\n"
            joined = sep.join(t.strip() for t in texts if t)
            total_tokens = len(tokenizer.encode(joined, add_special_tokens=False))

            # If it fits, single pass with long target
            final_target = self._choose_final_new_tokens(total_tokens, dec_cap)
            if total_tokens <= src_cap:
                return self._summ_text(
                    model,
                    tokenizer,
                    device,
                    joined,
                    src_cap,
                    self._gen_args_long(
                        final_target, max(80, final_target // 7), tokenizer
                    ),
                )

            # Otherwise: pack by docs → per-chunk → reduce
            # src_cap - 100 --> encoder’s safe capacity + Reserves 100 tokens for special tokens or extra safety margin.
            chunk_tokens = max(768, min(1200, src_cap - 100))
            parts = self._pack_by_docs_with_budget(
                texts, metas, tokenizer, chunk_tokens, sep_doc=sep
            )
            n = len(parts)
            per_chunk_target = self._choose_per_chunk_new_tokens(n, final_target)

            # Optionally weight by chunk size (simple uniform is usually fine):
            # per_chunk_targets = [max(60, min(220, int(per_chunk_target * (len(tokenizer.encode(p, add_special_tokens=False)) / total_tokens)))) for p in parts]
            # For simplicity, use uniform target:
            per_gen = self._gen_args_chunk(per_chunk_target, tokenizer)

            partials = [
                self._summ_text(model, tokenizer, device, p, src_cap, per_gen)
                for p in parts
            ]

            combined = "\n\n### CHUNK SUMMARY ###\n\n".join(partials)
            return self._summ_text(
                model,
                tokenizer,
                device,
                combined,
                src_cap,
                self._gen_args_long(
                    final_target, max(80, final_target // 7), tokenizer
                ),
            )
        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Error summarizing cluster: {e}"
            )
            raise e

    def _safe_caps(self, model, tokenizer) -> tuple[int, int]:
        """
        how many tokens you can safely give to the encoder and decoder of  model without breaking its limits.
        src_cap → Max tokens for the source (encoder input).
        dec_cap → Max tokens for the target (decoder output).
        """
        cap = int(getattr(model.config, "max_position_embeddings", 1024))
        tok_cap = int(getattr(tokenizer, "model_max_length", cap))
        if tok_cap > 1_000_000:  # ---> tokenizer "infinite" sentinel
            tok_cap = cap
        src_cap = max(32, min(cap, tok_cap) - 4)  # ----> encoder budget
        dec_cap = cap - 4  # ---> decoder budget
        return src_cap, dec_cap

    def _choose_final_new_tokens(self, total_tokens: int, dec_cap: int) -> int:
        """
        Piecewise schedule: larger clusters → longer final summary.
        Clamped to decoder cap.
        """
        # if total_tokens <= 800:    tgt = 240
        # elif total_tokens <= 1600: tgt = 360
        # elif total_tokens <= 3200: tgt = 480
        # elif total_tokens <= 6400: tgt = 600
        # else:                      tgt = 700
        tgt = 700
        return int(min(tgt, dec_cap))

    @torch.inference_mode()
    def _summ_text(
        self, model, tokenizer, device, text: str, src_cap: int, gen: dict
    ) -> str:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=src_cap,
            padding=False,
        )
        out = model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            **gen,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def _gen_args_long(self, final_max_new: int, final_min_new: int, tokenizer) -> dict:
        return dict(
            num_beams=4,
            max_new_tokens=final_max_new,
            min_new_tokens=final_min_new,
            length_penalty=1.05 if final_max_new >= 600 else 1.1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    def _gen_args_chunk(self, per_max_new: int, tokenizer) -> dict:
        return dict(
            num_beams=4,
            max_new_tokens=per_max_new,
            min_new_tokens=min(60, per_max_new),
            length_penalty=1.6,  # keep chunks concise
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    def _pack_by_docs_with_budget(
        self, texts, metas, tokenizer, tokens_per_chunk: int, sep_doc="\n\n— DOC —\n\n"
    ):
        """
        Pack documents into chunks, respecting budget.
        - texts: List of document texts.
        - metas: List of document metadata dictionaries.
        - tokens_per_chunk: Maximum number of tokens per chunk.
        - sep_doc: Separator between documents.
        Returns: List of chunked documents.
        """
        chunks, cur_docs, cur_cost = [], [], 0
        sep_ids = tokenizer.encode(sep_doc, add_special_tokens=False)
        sep_cost = len(sep_ids)

        for t, m in zip(texts, metas):
            if not t:
                continue
            head = m.get("domain") or m.get("source") or ""
            doc_text = (f"[{head}] " if head else "") + t.strip()
            doc_ids = tokenizer.encode(doc_text, add_special_tokens=False)
            doc_len = len(doc_ids)

            extra = 0 if not cur_docs else sep_cost
            if cur_docs and cur_cost + extra + doc_len > tokens_per_chunk:
                chunks.append(sep_doc.join(cur_docs))
                cur_docs, cur_cost = [], 0

            if doc_len > tokens_per_chunk:  # single huge doc → hard clamp
                cur_docs.append(
                    tokenizer.decode(
                        doc_ids[:tokens_per_chunk], skip_special_tokens=True
                    )
                )
                chunks.append(sep_doc.join(cur_docs))
                cur_docs, cur_cost = [], 0
            else:
                cur_docs.append(doc_text)
                cur_cost += doc_len + extra

        if cur_docs:
            chunks.append(sep_doc.join(cur_docs))
        return chunks

    def _choose_per_chunk_new_tokens(self, n_chunks: int, final_new_tokens: int) -> int:
        """
        Allocate ~35% of final budget to the map stage, split across chunks,
        with sensible floor/ceiling.
        """
        if n_chunks <= 0:
            return 120
        pool = int(final_new_tokens * 0.35)
        per = max(60, min(220, pool // n_chunks))
        return per

    async def _initialize_inference_runtime(self) -> InferenceRuntime:
        """Initialize the inference runtime for summarization."""
        try:
            # Create runtime config optimized for summarization
            config = RuntimeConfig()

            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_manager_loader=lambda: SummarizationModelManager(self.settings),
                config=config,
            )

            await self.inference_runtime.start()
            self.logger.info(
                "cluster_summary_generator.py:ClusterSummaryGenerator:Inference runtime initialized for summarization"
            )

            return self.inference_runtime

        except Exception as e:
            self.logger.error(
                f"cluster_summary_generator.py:ClusterSummaryGenerator:Failed to initialize inference runtime: {e}"
            )
            raise
