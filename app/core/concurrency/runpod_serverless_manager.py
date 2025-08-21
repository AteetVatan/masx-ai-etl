# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
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
RunPod Serverless Manager for parallel execution.

This module manages distribution of flashpoints across multiple RunPod Serverless
instances for true parallel processing. Perfect for RunPod Serverless with
configurable max workers.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple, Optional
from math import sqrt
from app.config import get_service_logger
from app.etl_data.etl_models import FlashpointModel
from app.config import get_settings

logger = get_service_logger("RunPodServerlessManager")


class RunPodServerlessManager:
    """
    Manages distribution across RunPod Serverless instances.
    Perfect for RunPod Serverless with configurable max workers.
    """
    
    def __init__(self, num_workers: int = 1):
        self.num_workers = max(1, num_workers)        
        self.num_workers -= 1 # 1 is used for the coordinator
        
        self.logger = get_service_logger("RunPodServerlessManager")
        self.settings = get_settings()
        self.runpod_api_key = self.settings.runpod_api_key
        self.runpod_endpoint = self.settings.runpod_endpoint
        
        # Validate configuration
        if self.num_workers > 1 and not self.runpod_api_key:
            self.logger.warning("RUNPOD_API_KEY not set, falling back to single worker mode")
            self.num_workers = 1
        if self.num_workers > 1 and not self.runpod_endpoint:
            self.logger.warning("RUNPOD_ENDPOINT not set, falling back to single worker mode")
            self.num_workers = 1
        
    from typing import List, Tuple

    def _distribute_flashpoints(self, flashpoints: List["FlashpointModel"]) -> List[List["FlashpointModel"]]:
        """
        Distribute flashpoints across workers by *workload* (number of feeds), not just count.

        Strategy:
        1) Sort flashpoints by feeds_count desc.
        2) Assign top-K (K = min(num_workers, N)) heavy flashpoints as singletons (one per worker).
        3) Assign remaining flashpoints greedily to the worker with the lowest cumulative feeds_count.
            (LPT - Longest Processing Time heuristic)

        Returns:
        List[List[FlashpointModel]] of length exactly self.num_workers.
        """
        num_workers = self.num_workers
        if not flashpoints:
            return [[] for _ in range(num_workers)]
        
        if num_workers == 1 or len(flashpoints) == 1:
            return [flashpoints]

        # Precompute feed counts (robust to None/missing)
        def feed_count(fp: FlashpointModel) -> int:
            fp.feeds
            feeds = fp.feeds
            try:
                return len(feeds)
            except Exception:
                return 0

        flashpoint_feeds_tuple_list: List[Tuple["FlashpointModel", int]] = [(fp, feed_count(fp)) for fp in flashpoints]

        # Log inventory
        for fp, c in flashpoint_feeds_tuple_list:            
            self.logger.info(f"Flashpoint {fp.id}: {fp.title} | feeds={c}")

        # Sort heavy → light; tie-break deterministically by id if present
        flashpoint_feeds_tuple_list.sort(key=lambda t: (-t[1], str(getattr(t[0], "id", ""))))

        # Initialize worker bins
        chunks: List[List["FlashpointModel"]] = [[] for _ in range(num_workers)]
        loads: List[int] = [0] * num_workers

        # Phase 1: dedicate the top-K heavy flashpoints one-per-worker
        K = min(num_workers, len(flashpoint_feeds_tuple_list))
        for i in range(K):
            fp, c = flashpoint_feeds_tuple_list[i]
            chunks[i].append(fp)
            loads[i] += c

        # Phase 2: greedily place the rest onto the least-loaded worker
        for fp, c in flashpoint_feeds_tuple_list[K:]:
            idx = min(range(num_workers), key=lambda i: loads[i])
            chunks[idx].append(fp)
            loads[idx] += c

        # Final logs
        total_feeds = sum(c for _, c in flashpoint_feeds_tuple_list)
        self.logger.info(
            f"Distributed {len(flashpoints)} flashpoints "
            f"(total_feeds={total_feeds}) across {num_workers} workers"
        )
        for i, (chunk, load) in enumerate(zip(chunks, loads), start=1):
            self.logger.info(f"Worker {i}: {len(chunk)} flashpoints, sum_feeds={load}")

        return chunks

    
    async def distribute_to_workers(
        self, 
        flashpoints: List[FlashpointModel], 
        date: str,
        cleanup: bool = True
    ) -> List[Any]:
        """
        Distribute flashpoints to multiple RunPod Serverless instances.
        
        Args:
            flashpoints: List of flashpoints to process
            date: Date for processing
            cleanup: Whether to cleanup before processing
            
        Returns:
            List of results from all workers
        """
        if self.num_workers == 1:
            # Single worker - process locally
            self.logger.info("Single worker mode - processing locally")
            return await self._process_locally(flashpoints, date, cleanup)
        
        # Split flashpoints into chunks
        worker_chunks = self._distribute_flashpoints(flashpoints)
        
        # Send each chunk to a different RunPod instance
        worker_tasks = []
        for i, chunk in enumerate(worker_chunks):
            if chunk:
                task = self._send_to_worker_instance(i + 1, chunk, date, cleanup)
                worker_tasks.append(task)
        
        # Wait for all workers to complete
        if worker_tasks:
            self.logger.info(f"Starting {len(worker_tasks)} RunPod Serverless workers")
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            return self._aggregate_results(results, flashpoints)
        
        return []
    
    async def _send_to_worker_instance(
        self, 
        worker_id: int, 
        flashpoints: List[FlashpointModel], 
        date: str, 
        cleanup: bool
    ) -> Dict[str, Any]:
        """
        Send work to a RunPod Serverless instance.
        
        Args:
            worker_id: ID of the worker
            flashpoints: Flashpoints for this worker
            date: Date for processing
            cleanup: Whether to cleanup
            
        Returns:
            Result from the worker instance
        """
        # Create payload for this worker
        payload = {
            "input": {
                "date": date,
                "trigger": "etl_worker",
                "cleanup": cleanup,
                "flashpoints": [fp.id for fp in flashpoints]
            },
            "policy": {
                "executionTimeout": 10800000,
                "ttl": 14400000
            }
        }
        
        self.logger.info(f"Worker {worker_id}: Sending {len(flashpoints)} flashpoints to RunPod")
        
        # Call RunPod API to create new instance
        # Note: No timeout set - ETL processes can take hours to complete
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.runpod_api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                # This creates a new RunPod Serverless instance
                async with session.post(
                    self.runpod_endpoint,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Worker {worker_id} instance created successfully")
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"RunPod API error {response.status}: {error_text}")
                        
            except Exception as e:
                self.logger.error(f"Error creating worker {worker_id}: {e}")
                raise e
    
    async def _process_locally(
        self, 
        flashpoints: List[FlashpointModel], 
        date: str, 
        cleanup: bool
    ) -> List[Any]:
        """
        Process flashpoints locally (single worker mode).
        
        Args:
            flashpoints: List of flashpoints to process
            date: Date for processing
            cleanup: Whether to cleanup
            
        Returns:
            List of results
        """
        self.logger.info(f"Processing {len(flashpoints)} flashpoints locally")
        
        # Import here to avoid circular imports
        from app.etl import ETLPipeline
        
        # Create ETL pipeline and process
        etl_pipeline = ETLPipeline(date)
        results = []
        
        for flashpoint in flashpoints:
            try:
                result = await etl_pipeline.run_etl_pipeline(flashpoint)
                results.append(result)
                self.logger.info(f"Completed flashpoint {flashpoint.id}")
            except Exception as e:
                self.logger.error(f"Error processing flashpoint {flashpoint.id}: {e}")
                results.append(None)
        
        return results
    
    def _aggregate_results(
        self, 
        worker_results: List[Any], 
        original_flashpoints: List[FlashpointModel]
    ) -> List[Any]:
        """
        Aggregate results from multiple workers.
        
        Args:
            worker_results: Results from all workers
            original_flashpoints: Original flashpoints list
            
        Returns:
            Flattened list of results
        """
        self.logger.info("Aggregating results from all workers")
        
        # Flatten results and handle exceptions
        flattened_results = []
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                self.logger.error(f"Worker {i+1} error: {result}")
                # Estimate how many results this worker should have produced
                estimated_results = len(original_flashpoints) // self.num_workers
                flattened_results.extend([None] * estimated_results)
            else:
                # Add the worker's results
                if isinstance(result, list):
                    flattened_results.extend(result)
                else:
                    flattened_results.append(result)
        
        self.logger.info(f"Aggregated {len(flattened_results)} results from {len(worker_results)} workers")
        return flattened_results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the manager.
        
        Returns:
            Status information
        """
        return {
            "num_workers": self.num_workers,
            "runpod_api_key_configured": bool(self.runpod_api_key),
            "runpod_endpoint_configured": bool(self.runpod_endpoint),
            "mode": "multi_worker" if self.num_workers > 1 else "single_worker"
        }
