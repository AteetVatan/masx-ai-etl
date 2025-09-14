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
Migration guide from singleton ModelManager to new model management system.

This module provides step-by-step examples of how to migrate existing code
from the singleton ModelManager to the new instance-based model managers.
"""

from app.core.models import SummarizationModelManager, EmbeddingModelManager
from app.config import get_settings


# ============================================================================
# MIGRATION EXAMPLES
# ============================================================================

def migration_example_1_basic_usage():
    """
    Migration Example 1: Basic Model Usage
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    
    # Get model components
    model, tokenizer, device = ModelManager.get_summarization_model()
    
    # Use model directly
    # ... model inference code ...
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import SummarizationModelManager
    from app.config import get_settings
    
    # Create manager instance
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Use with context manager (recommended)
    with summarizer.acquire() as instance:
        summary = summarizer.summarize_text(text, instance)
    ```
    """
    print("=== Migration Example 1: Basic Usage ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # model, tokenizer, device = ModelManager.get_summarization_model()
    
    # NEW WAY
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    text = "This is a sample article to summarize..."
    
    # Use context manager (recommended approach)
    with summarizer.acquire() as instance:
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary: {summary}")


def migration_example_2_embedding_usage():
    """
    Migration Example 2: Embedding Model Usage
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    
    # Get embedding model
    embedding_model = ModelManager.get_embedding_model()
    
    # Use model directly
    embeddings = embedding_model.encode(texts)
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import EmbeddingModelManager
    from app.config import get_settings
    
    # Create manager instance
    settings = get_settings()
    embedder = EmbeddingModelManager(settings)
    
    # Use with context manager
    with embedder.acquire() as instance:
        embeddings = embedder.encode_texts(texts, instance)
    ```
    """
    print("\n=== Migration Example 2: Embedding Usage ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # embedding_model = ModelManager.get_embedding_model()
    # embeddings = embedding_model.encode(texts)
    
    # NEW WAY
    settings = get_settings()
    embedder = EmbeddingModelManager(settings)
    
    texts = ["Document 1", "Document 2", "Document 3"]
    
    # Use context manager
    with embedder.acquire() as instance:
        embeddings = embedder.encode_texts(texts, instance, convert_to_tensor=False)
        print(f"Encoded {len(texts)} texts, embedding dimension: {len(embeddings[0])}")


def migration_example_3_explicit_acquire_release():
    """
    Migration Example 3: Explicit Acquire/Release Pattern
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    
    # Models are always available (singleton)
    model, tokenizer, device = ModelManager.get_summarization_model()
    # ... use model ...
    # No cleanup needed (singleton persists)
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import SummarizationModelManager
    from app.config import get_settings
    
    # Create manager
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Explicit acquire/release
    instance = summarizer.get_instance()
    try:
        # ... use instance ...
        summary = summarizer.summarize_text(text, instance)
    finally:
        summarizer.release_instance(instance)
    ```
    """
    print("\n=== Migration Example 3: Explicit Acquire/Release ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # model, tokenizer, device = ModelManager.get_summarization_model()
    # # ... use model ...
    
    # NEW WAY
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    text = "Another article to process..."
    
    # Explicit acquire/release pattern
    instance = summarizer.get_instance(timeout=10.0)
    try:
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary: {summary}")
    finally:
        summarizer.release_instance(instance)


def migration_example_4_concurrent_processing():
    """
    Migration Example 4: Concurrent Processing
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    import threading
    
    def worker():
        # All threads share the same singleton model
        model, tokenizer, device = ModelManager.get_summarization_model()
        # ... process ...
    
    # Start multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import SummarizationModelManager
    from app.config import get_settings
    import threading
    
    def worker():
        # Each thread gets its own model instance from the pool
        settings = get_settings()
        summarizer = SummarizationModelManager(settings)
        
        with summarizer.acquire() as instance:
            # ... process with instance ...
            summary = summarizer.summarize_text(text, instance)
    
    # Start multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    ```
    """
    print("\n=== Migration Example 4: Concurrent Processing ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # import threading
    # 
    # def worker():
    #     model, tokenizer, device = ModelManager.get_summarization_model()
    #     # ... process ...
    # 
    # threads = []
    # for i in range(5):
    #     thread = threading.Thread(target=worker)
    #     threads.append(thread)
    #     thread.start()
    
    # NEW WAY
    import threading
    import time
    
    def worker(worker_id: int, text: str):
        settings = get_settings()
        summarizer = SummarizationModelManager(settings)
        
        with summarizer.acquire() as instance:
            summary = summarizer.summarize_text(text, instance)
            print(f"Worker {worker_id}: {summary[:30]}...")
    
    # Start multiple threads
    threads = []
    for i in range(3):
        text = f"Article {i+1} about machine learning and AI..."
        thread = threading.Thread(target=worker, args=(i, text))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()


def migration_example_5_memory_management():
    """
    Migration Example 5: Memory Management
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    
    # Models are loaded once and persist in memory
    model, tokenizer, device = ModelManager.get_summarization_model()
    # ... use model ...
    # No explicit memory management
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import SummarizationModelManager
    from app.config import get_settings
    
    # Create manager
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Use with automatic cleanup
    with summarizer.acquire(destroy_after_use=True) as instance:
        summary = summarizer.summarize_text(text, instance)
    
    # Or manual cleanup
    summarizer.cleanup()  # Destroy all instances
    ```
    """
    print("\n=== Migration Example 5: Memory Management ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # model, tokenizer, device = ModelManager.get_summarization_model()
    # # ... use model ...
    # # No explicit cleanup
    
    # NEW WAY
    settings = get_settings()
    summarizer = SummarizationModelManager(settings)
    
    # Method 1: Automatic cleanup with context manager
    with summarizer.acquire(destroy_after_use=True) as instance:
        text = "Memory-intensive processing task..."
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary with auto-cleanup: {summary[:30]}...")
    
    # Method 2: Manual cleanup
    instance = summarizer.get_instance()
    try:
        text = "Another processing task..."
        summary = summarizer.summarize_text(text, instance)
        print(f"Summary with manual cleanup: {summary[:30]}...")
    finally:
        summarizer.release_instance(instance, destroy=True)
    
    # Method 3: Full cleanup
    summarizer.cleanup()
    print("All resources cleaned up")


def migration_example_6_class_based_usage():
    """
    Migration Example 6: Class-based Usage
    
    BEFORE (Singleton Pattern):
    ```python
    from app.singleton import ModelManager
    
    class Summarizer:
        def __init__(self):
            self.model, self.tokenizer, self.device = ModelManager.get_summarization_model()
        
        def summarize(self, text):
            # ... use self.model, self.tokenizer, self.device ...
    ```
    
    AFTER (Instance-based Pattern):
    ```python
    from app.core.models import SummarizationModelManager
    from app.config import get_settings
    
    class Summarizer:
        def __init__(self):
            settings = get_settings()
            self.manager = SummarizationModelManager(settings)
        
        def summarize(self, text):
            with self.manager.acquire() as instance:
                return self.manager.summarize_text(text, instance)
    ```
    """
    print("\n=== Migration Example 6: Class-based Usage ===")
    
    # OLD WAY (commented out for reference)
    # from app.singleton import ModelManager
    # 
    # class Summarizer:
    #     def __init__(self):
    #         self.model, self.tokenizer, self.device = ModelManager.get_summarization_model()
    #     
    #     def summarize(self, text):
    #         # ... use self.model, self.tokenizer, self.device ...
    
    # NEW WAY
    class Summarizer:
        def __init__(self):
            settings = get_settings()
            self.manager = SummarizationModelManager(settings)
        
        def summarize(self, text: str) -> str:
            with self.manager.acquire() as instance:
                return self.manager.summarize_text(text, instance)
    
    # Usage
    summarizer = Summarizer()
    text = "Class-based summarization example..."
    summary = summarizer.summarize(text)
    print(f"Class-based summary: {summary[:30]}...")


# ============================================================================
# MIGRATION CHECKLIST
# ============================================================================

def print_migration_checklist():
    """Print a comprehensive migration checklist."""
    print("\n" + "="*60)
    print("MIGRATION CHECKLIST")
    print("="*60)
    
    checklist = [
        "1. Update imports:",
        "   - Replace: from app.singleton import ModelManager",
        "   - With: from app.core.models import SummarizationModelManager, EmbeddingModelManager",
        "",
        "2. Create manager instances:",
        "   - Add: settings = get_settings()",
        "   - Add: manager = SummarizationModelManager(settings)",
        "",
        "3. Replace direct model access:",
        "   - Replace: model, tokenizer, device = ModelManager.get_summarization_model()",
        "   - With: with manager.acquire() as instance: ...",
        "",
        "4. Update method calls:",
        "   - Replace direct model usage with manager methods",
        "   - Use: manager.summarize_text(text, instance)",
        "   - Use: manager.encode_texts(texts, instance)",
        "",
        "5. Handle concurrency:",
        "   - Each thread/process should create its own manager instance",
        "   - Use context managers for automatic cleanup",
        "",
        "6. Add error handling:",
        "   - Handle TimeoutError for acquire operations",
        "   - Use try/finally for explicit acquire/release",
        "",
        "7. Memory management:",
        "   - Use destroy_after_use=True for memory-intensive tasks",
        "   - Call manager.cleanup() when done",
        "",
        "8. Testing:",
        "   - Update tests to use new managers",
        "   - Mock the managers instead of singleton",
        "",
        "9. Configuration:",
        "   - Update settings.py with model_pool_max_instances",
        "   - Configure GPU/CPU settings as needed",
        "",
        "10. Performance monitoring:",
        "    - Use manager.get_pool_stats() to monitor usage",
        "    - Adjust pool sizes based on workload",
    ]
    
    for item in checklist:
        print(item)


def main():
    """Run all migration examples."""
    print("MASX AI Model Manager Migration Guide")
    print("=" * 50)
    
    try:
        # Run migration examples
        migration_example_1_basic_usage()
        migration_example_2_embedding_usage()
        migration_example_3_explicit_acquire_release()
        migration_example_4_concurrent_processing()
        migration_example_5_memory_management()
        migration_example_6_class_based_usage()
        
        # Print migration checklist
        print_migration_checklist()
        
    except Exception as e:
        print(f"Error running migration examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
