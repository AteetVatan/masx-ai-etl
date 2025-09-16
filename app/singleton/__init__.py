"""Init file for the singleton package."""

from .proxy_manager import ProxyManager
from .chroma_client_singleton import ChromaClientSingleton


# Lazy import for ProxyManager to avoid circular dependency
def get_proxy_manager():
    """Lazy import to avoid circular dependency."""
    from .proxy_manager import ProxyManager

    return ProxyManager


__all__ = [
    "ChromaClientSingleton",
]
