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
from app.enumeration import WorkerEnums

logger = get_service_logger("RunPodServerlessManager")


class RunPodServerlessManager:
    """
    Manages distribution across RunPod Serverless instances.
    Perfect for RunPod Serverless with configurable max workers.
    """

    def __init__(self, num_workers: int = 1):
        self.num_workers = max(1, num_workers)
        self.num_workers -= 1  # 1 is used for the coordinator

        self.logger = get_service_logger("RunPodServerlessManager")
        self.settings = get_settings()
        self.runpod_api_key = self.settings.runpod_api_key
        self.runpod_endpoint = self.settings.runpod_endpoint

        # Validate configuration
        if self.num_workers > 1 and not self.runpod_api_key:
            self.logger.warning(
                "runpod_serverless_manager.py:RunPodServerlessManager:RUNPOD_API_KEY not set, falling back to single worker mode"
            )
            self.num_workers = 1
        if self.num_workers > 1 and not self.runpod_endpoint:
            self.logger.warning(
                "runpod_serverless_manager.py:RunPodServerlessManager:RUNPOD_ENDPOINT not set, falling back to single worker mode"
            )
            self.num_workers = 1

    from typing import List, Tuple

    def _distribute_flashpoints(
        self, flashpoints: List["FlashpointModel"]
    ) -> List[List["FlashpointModel"]]:
        """
        Distribute flashpoints across workers: 1 flashpoint per worker.

        Strategy:
        1) Assign one flashpoint per worker in round-robin fashion
        2) If more flashpoints than workers, distribute evenly
        3) If fewer flashpoints than workers, some workers will be idle

        Returns:
        List[List[FlashpointModel]] of length exactly self.num_workers.
        """
        num_workers = self.num_workers
        if not flashpoints:
            return [[] for _ in range(num_workers)]

        if num_workers == 1:
            return [flashpoints]
        
        
        #########################################################
        # for now we are not using the round-robin distribution
        # one flashpoint per worker
        num_workers = len(flashpoints) 
        
        #########################################################

        # Initialize worker bins
        chunks: List[List["FlashpointModel"]] = [[] for _ in range(num_workers)]

        # Simple round-robin distribution: 1 flashpoint per worker
        

     
        
        
        for i, flashpoint in enumerate(flashpoints):
            worker_index = i % num_workers
            chunks[worker_index].append(flashpoint)

        # Log distribution
        self.logger.info(
            f"runpod_serverless_manager.py:RunPodServerlessManager:Distributed {len(flashpoints)} flashpoints "
            f"across {num_workers} workers (1 per worker)"
        )

        for i, chunk in enumerate(chunks, start=1):
            if chunk:
                self.logger.info(
                    f"runpod_serverless_manager.py:RunPodServerlessManager:Worker {i}: {len(chunk)} flashpoint(s)"
                )
            else:
                self.logger.info(
                    f"runpod_serverless_manager.py:RunPodServerlessManager:Worker {i}: idle (no flashpoints assigned)"
                )

        return chunks

    async def distribute_to_workers(
        self, flashpoints: List[FlashpointModel], date: str, cleanup: bool = True
    ) -> List[Any]:
        """
        Distribute flashpoints to multiple RunPod Serverless instances (fire and forget).

        Args:
            flashpoints: List of flashpoints to process
            date: Date for processing
            cleanup: Whether to cleanup before processing

        Returns:
            Status information about launched workers (no actual results)
        """
        if self.num_workers == 1:
            # Single worker - process locally
            self.logger.info(
                "runpod_serverless_manager.py:RunPodServerlessManager:Single worker mode - processing locally"
            )
            return await self._process_locally(flashpoints, date, cleanup)

        # Split flashpoints into chunks
        worker_chunks = self._distribute_flashpoints(flashpoints)

        # Send each chunk to a different RunPod instance (fire and forget)
        launched_workers = 0
        
        
        for i, chunk in enumerate(worker_chunks):
            if chunk:
                self.logger.info(
                    f"runpod_serverless_manager.py:RunPodServerlessManager:**********Launching chunk {i + 1} to RunPod Serverless worker (fire and forget)***********"
                )
                # Fire and forget - don't wait for completion
                asyncio.create_task(
                    self._send_to_worker_instance(i + 1, chunk, date, cleanup)
                )
                launched_workers += 1

        if launched_workers > 0:
            self.logger.info(
                f"runpod_serverless_manager.py:RunPodServerlessManager:Launched {launched_workers} RunPod Serverless workers (fire and forget mode)"
            )
            # Return immediately - workers will process independently
            return [
                {
                    "status": "workers_launched",
                    "count": launched_workers,
                    "mode": "fire_and_forget",
                }
            ]

        return []

    async def _send_to_worker_instance(
        self,
        worker_id: int,
        flashpoints: List[FlashpointModel],
        date: str,
        cleanup: bool,
    ) -> None:
        """
        Send work to a RunPod Serverless instance (fire and forget).

        Args:
            worker_id: ID of the worker
            flashpoints: Flashpoints for this worker
            date: Date for processing
            cleanup: Whether to cleanup

        Returns:
            None (fire and forget - no results returned)
        """
        # Create payload for this worker
        payload = {
            "input": {
                "date": date,
                "trigger": WorkerEnums.ETL_WORKER.value,
                "cleanup": cleanup,
                "flashpoints": [fp.id for fp in flashpoints],
            },
            "policy": {"executionTimeout": 10800000, "ttl": 14400000},
        }

        self.logger.info(
            f"runpod_serverless_manager.py:RunPodServerlessManager:Worker {worker_id}: Launching {len(flashpoints)} flashpoints to RunPod (fire and forget)"
        )

        # Call RunPod API to create new instance (fire and forget)
        # Note: No timeout set - ETL processes can take hours to complete
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.runpod_api_key}",
                    "Content-Type": "application/json",
                }

                # This creates a new RunPod Serverless instance
                async with session.post(
                    self.runpod_endpoint, json=payload, headers=headers
                ) as response:
                    if response.status == 200:
                        self.logger.info(
                            f"runpod_serverless_manager.py:RunPodServerlessManager:Worker {worker_id} instance launched successfully"
                        )
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"runpod_serverless_manager.py:RunPodServerlessManager:Worker {worker_id} RunPod API error {response.status}: {error_text}"
                        )

        except Exception as e:
            self.logger.error(
                f"runpod_serverless_manager.py:RunPodServerlessManager:Error launching worker {worker_id}: {e}"
            )
            # Don't raise - this is fire and forget, just log the error

    async def _process_locally(
        self, flashpoints: List[FlashpointModel], date: str, cleanup: bool
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
        self.logger.info(
            f"runpod_serverless_manager.py:RunPodServerlessManager:Processing {len(flashpoints)} flashpoints locally"
        )

        # Import here to avoid circular imports
        from app.etl import ETLPipeline

        # Create ETL pipeline and process
        etl_pipeline = ETLPipeline(date)
        results = []

        for flashpoint in flashpoints:
            try:
                result = await etl_pipeline.run_etl_pipeline(flashpoint)
                results.append(result)
                self.logger.info(
                    f"runpod_serverless_manager.py:RunPodServerlessManager:Completed flashpoint {flashpoint.id}"
                )
            except Exception as e:
                self.logger.error(
                    f"runpod_serverless_manager.py:RunPodServerlessManager:Error processing flashpoint {flashpoint.id}: {e}"
                )
                results.append(None)

        return results

    def _aggregate_results(
        self, worker_results: List[Any], original_flashpoints: List[FlashpointModel]
    ) -> List[Any]:
        """
        Aggregate results from multiple workers (legacy method - not used in fire and forget mode).

        Args:
            worker_results: Results from all workers
            original_flashpoints: Original flashpoints list

        Returns:
            Flattened list of results
        """
        self.logger.warning(
            "runpod_serverless_manager.py:RunPodServerlessManager:_aggregate_results called but not used in fire and forget mode"
        )
        return []

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
            "mode": (
                "multi_worker_fire_forget" if self.num_workers > 1 else "single_worker"
            ),
            "execution_strategy": (
                "fire_and_forget" if self.num_workers > 1 else "local_processing"
            ),
        }
