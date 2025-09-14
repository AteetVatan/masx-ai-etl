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
Integration examples for the new model management system.

This module demonstrates how to use the new model managers with both
context manager and explicit acquire/release patterns, including
concurrent usage and memory management.
"""

import asyncio
import threading
import time
from typing import List

from app.core.models import SummarizationModelManager, EmbeddingModelManager
from app.config import get_settings


def example_context_manager_usage():
    """Example: Using context manager pattern (recommended)."""
    print("=== Context Manager Usage Example ===")
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Context manager automatically handles acquire/release
    with summarizer.acquire() as instance:
        text = "This is a long article about artificial intelligence and machine learning..."
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary: {summary}")
    
    # Pool stats after use
    stats = summarizer.get_pool_stats()
    print(f"Pool stats: {stats}")


def example_explicit_acquire_release():
    """Example: Using explicit acquire/release pattern."""
    print("\n=== Explicit Acquire/Release Example ===")
    
    settings = get_settings()
    embedder = EmbeddingModelManager(settings)
    
    # Explicit acquire
    instance = embedder.get_instance(timeout=10.0)
    try:
        texts = ["First document", "Second document", "Third document"]
        embeddings = embedder.encode_texts(texts, instance, convert_to_tensor=False)
        print(f"Encoded {len(texts)} texts, embedding dimension: {len(embeddings[0])}")
    finally:
        # Always release
        embedder.release_instance(instance)
    
    # Pool stats after use
    stats = embedder.get_pool_stats()
    print(f"Pool stats: {stats}")


def example_destroy_after_use():
    """Example: Destroying instances after use to free memory."""
    print("\n=== Destroy After Use Example ===")
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Use context manager with destroy_after_use=True
    with summarizer.acquire(destroy_after_use=True) as instance:
        text = "Another article about natural language processing..."
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary: {summary}")
    
    # Pool stats after destruction
    stats = summarizer.get_pool_stats()
    print(f"Pool stats after destruction: {stats}")


def example_concurrent_usage():
    """Example: Concurrent usage with multiple threads."""
    print("\n=== Concurrent Usage Example ===")
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    results = []
    errors = []
    
    def worker(worker_id: int, texts: List[str]):
        """Worker function for concurrent processing."""
        try:
            for i, text in enumerate(texts):
                with summarizer.acquire(timeout=5.0) as instance:
                    summary = summarizer.summarize_text(text, instance)
                    result = f"Worker {worker_id}, Text {i}: {summary[:50]}..."
                    results.append(result)
                    time.sleep(0.1)  # Simulate processing time
        except Exception as e:
            errors.append(f"Worker {worker_id} error: {e}")
    
    # Prepare test data
    texts = [
        "Article about machine learning algorithms and their applications.",
        "Research paper on deep learning neural networks.",
        "Blog post about natural language processing techniques.",
        "Technical documentation on transformer models.",
    ]
    
    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i, texts))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Print results
    print(f"Completed {len(results)} tasks")
    print(f"Errors: {len(errors)}")
    if errors:
        for error in errors:
            print(f"  {error}")
    
    # Final pool stats
    stats = summarizer.get_pool_stats()
    print(f"Final pool stats: {stats}")


async def example_async_usage():
    """Example: Async usage pattern."""
    print("\n=== Async Usage Example ===")
    
    settings = get_settings()
    embedder = EmbeddingModelManager(settings)
    
    async def async_worker(worker_id: int, texts: List[str]):
        """Async worker function."""
        try:
            with embedder.acquire(timeout=5.0) as instance:
                embeddings = embedder.encode_texts(texts, instance, convert_to_tensor=False)
                return f"Worker {worker_id}: encoded {len(texts)} texts"
        except Exception as e:
            return f"Worker {worker_id} error: {e}"
    
    # Prepare test data
    texts = [
        "Document about artificial intelligence",
        "Paper on machine learning",
        "Article about deep learning",
    ]
    
    # Run async tasks
    tasks = []
    for i in range(3):
        task = async_worker(i, texts)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Print results
    for result in results:
        print(result)
    
    # Pool stats
    stats = embedder.get_pool_stats()
    print(f"Pool stats: {stats}")


def example_pool_management():
    """Example: Pool management and memory optimization."""
    print("\n=== Pool Management Example ===")
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Initial stats
    stats = summarizer.get_pool_stats()
    print(f"Initial pool stats: {stats}")
    
    # Use multiple instances
    instances = []
    for i in range(3):
        instance = summarizer.get_instance(timeout=5.0)
        instances.append(instance)
        print(f"Acquired instance {i+1}")
    
    # Check stats with instances in use
    stats = summarizer.get_pool_stats()
    print(f"Stats with instances in use: {stats}")
    
    # Release instances
    for i, instance in enumerate(instances):
        summarizer.release_instance(instance)
        print(f"Released instance {i+1}")
    
    # Check stats after release
    stats = summarizer.get_pool_stats()
    print(f"Stats after release: {stats}")
    
    # Shrink pool to free memory
    destroyed = summarizer.shrink_pool(1)
    print(f"Destroyed {destroyed} instances to shrink pool")
    
    # Final stats
    stats = summarizer.get_pool_stats()
    print(f"Final pool stats: {stats}")
    
    # Cleanup
    summarizer.cleanup()
    print("Cleaned up all resources")


def example_gpu_memory_optimization():
    """Example: GPU memory optimization with dynamic pool sizing."""
    print("\n=== GPU Memory Optimization Example ===")
    
    settings = get_settings()
    # Force GPU usage for this example
    settings.masx_force_gpu = True
    settings.masx_force_cpu = False
    
    summarizer = SummarizationModelManager(settings)
    
    # Initialize and check pool size
    summarizer.initialize()
    stats = summarizer.get_pool_stats()
    print(f"GPU pool size: {stats['max_instances']}")
    
    # Simulate heavy workload
    print("Simulating heavy workload...")
    for i in range(5):
        with summarizer.acquire(destroy_after_use=True) as instance:
            text = f"Heavy processing task {i+1} with lots of text to summarize..."
            summary = summarizer.summarize_text(text, instance)
            print(f"Task {i+1} completed: {summary[:30]}...")
    
    # Final cleanup
    summarizer.cleanup()
    print("GPU memory optimization completed")


def main():
    """Run all examples."""
    print("MASX AI Model Manager Usage Examples")
    print("=" * 50)
    
    try:
        # Basic usage examples
        example_context_manager_usage()
        example_explicit_acquire_release()
        example_destroy_after_use()
        
        # Advanced usage examples
        example_concurrent_usage()
        example_pool_management()
        
        # GPU optimization example
        example_gpu_memory_optimization()
        
        # Async example
        print("\n=== Running Async Example ===")
        asyncio.run(example_async_usage())
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
