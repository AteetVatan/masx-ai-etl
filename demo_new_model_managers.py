#!/usr/bin/env python3
"""
Demonstration script for the new model management system.

This script shows the key features of the new model managers including:
- Context manager usage
- Explicit acquire/release patterns
- Concurrent processing
- Memory management
- Pool statistics
"""

import time
import threading
from typing import List

# Import the new model managers
from app.core.models import SummarizationModelManager, EmbeddingModelManager
from app.config import get_settings


def demo_basic_usage():
    """Demonstrate basic usage with context managers."""
    print("🔧 Basic Usage Demo")
    print("-" * 30)
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Context manager usage (recommended)
    with summarizer.acquire() as instance:
        text = "This is a sample article about artificial intelligence and machine learning technologies."
        summary = summarizer.summarize_text(text, instance)
        print(f"📝 Summary: {summary}")
    
    # Show pool statistics
    stats = summarizer.get_pool_stats()
    print(f"📊 Pool stats: {stats}")


def demo_explicit_acquire_release():
    """Demonstrate explicit acquire/release pattern."""
    print("\n🔧 Explicit Acquire/Release Demo")
    print("-" * 40)
    
    settings = get_settings()
    embedder = EmbeddingModelManager(settings)
    
    # Explicit acquire/release
    instance = embedder.get_instance(timeout=5.0)
    try:
        texts = ["Document about AI", "Paper on ML", "Article about NLP"]
        embeddings = embedder.encode_texts(texts, instance, convert_to_tensor=False)
        print(f"🔢 Encoded {len(texts)} texts, embedding dimension: {len(embeddings[0])}")
    finally:
        embedder.release_instance(instance)
    
    # Show pool statistics
    stats = embedder.get_pool_stats()
    print(f"📊 Pool stats: {stats}")


def demo_concurrent_processing():
    """Demonstrate concurrent processing with multiple threads."""
    print("\n🔧 Concurrent Processing Demo")
    print("-" * 35)
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    results = []
    errors = []
    
    def worker(worker_id: int, text: str):
        """Worker function for concurrent processing."""
        try:
            with summarizer.acquire(timeout=5.0) as instance:
                summary = summarizer.summarize_text(text, instance)
                result = f"Worker {worker_id}: {summary[:40]}..."
                results.append(result)
                time.sleep(0.1)  # Simulate processing time
        except Exception as e:
            errors.append(f"Worker {worker_id} error: {e}")
    
    # Prepare test data
    texts = [
        "Article about machine learning algorithms and their applications in various industries.",
        "Research paper on deep learning neural networks and their architecture.",
        "Blog post about natural language processing techniques and tools.",
        "Technical documentation on transformer models and attention mechanisms.",
    ]
    
    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i, texts[i]))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Print results
    print(f"✅ Completed {len(results)} tasks")
    for result in results:
        print(f"   {result}")
    
    if errors:
        print(f"❌ {len(errors)} errors occurred:")
        for error in errors:
            print(f"   {error}")
    
    # Final pool stats
    stats = summarizer.get_pool_stats()
    print(f"📊 Final pool stats: {stats}")


def demo_memory_management():
    """Demonstrate memory management and cleanup."""
    print("\n🔧 Memory Management Demo")
    print("-" * 30)
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Show initial stats
    stats = summarizer.get_pool_stats()
    print(f"📊 Initial pool stats: {stats}")
    
    # Use multiple instances
    instances = []
    for i in range(3):
        instance = summarizer.get_instance(timeout=5.0)
        instances.append(instance)
        print(f"🔒 Acquired instance {i+1}")
    
    # Check stats with instances in use
    stats = summarizer.get_pool_stats()
    print(f"📊 Stats with instances in use: {stats}")
    
    # Release instances
    for i, instance in enumerate(instances):
        summarizer.release_instance(instance)
        print(f"🔓 Released instance {i+1}")
    
    # Check stats after release
    stats = summarizer.get_pool_stats()
    print(f"📊 Stats after release: {stats}")
    
    # Shrink pool to free memory
    destroyed = summarizer.shrink_pool(1)
    print(f"🗑️  Destroyed {destroyed} instances to shrink pool")
    
    # Final stats
    stats = summarizer.get_pool_stats()
    print(f"📊 Final pool stats: {stats}")
    
    # Cleanup
    summarizer.cleanup()
    print("🧹 Cleaned up all resources")


def demo_pool_statistics():
    """Demonstrate pool statistics and monitoring."""
    print("\n🔧 Pool Statistics Demo")
    print("-" * 25)
    
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    embedder = EmbeddingModelManager(settings)
    
    # Show initial stats for both managers
    summarizer_stats = summarizer.get_pool_stats()
    embedder_stats = embedder.get_pool_stats()
    
    print(f"📊 Summarization pool: {summarizer_stats}")
    print(f"📊 Embedding pool: {embedder_stats}")
    
    # Use both managers
    with summarizer.acquire() as s_instance:
        with embedder.acquire() as e_instance:
            # Show stats while instances are in use
            summarizer_stats = summarizer.get_pool_stats()
            embedder_stats = embedder.get_pool_stats()
            
            print(f"📊 Summarization pool (in use): {summarizer_stats}")
            print(f"📊 Embedding pool (in use): {embedder_stats}")
    
    # Show final stats
    summarizer_stats = summarizer.get_pool_stats()
    embedder_stats = embedder.get_pool_stats()
    
    print(f"📊 Summarization pool (final): {summarizer_stats}")
    print(f"📊 Embedding pool (final): {embedder_stats}")


def main():
    """Run all demonstrations."""
    print("🚀 MASX AI Model Manager Demonstration")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_basic_usage()
        demo_explicit_acquire_release()
        demo_concurrent_processing()
        demo_memory_management()
        demo_pool_statistics()
        
        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Context manager usage (recommended)")
        print("✅ Explicit acquire/release patterns")
        print("✅ Concurrent processing with multiple threads")
        print("✅ Memory management and cleanup")
        print("✅ Pool statistics and monitoring")
        print("✅ Automatic GPU/CPU optimization")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
