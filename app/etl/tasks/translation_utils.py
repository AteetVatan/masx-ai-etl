
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Callable, Iterable, List, Tuple, TypeVar
import heapq

T = TypeVar("T")  # Your Feed model (e.g., FeedModel). We store only references, not text copies.

# ---------------------------
# Configs (tune with your telemetry)
# ---------------------------

@dataclass
class GoogleConfig:
    max_chars: int = 4800              # stay safely below the 5000 hard limit
    alpha_ms: float = 120.0            # fixed overhead per Google request (TLS, network)
    beta_ms_per_char: float = 0.010    # per-char time (measure!)

@dataclass
class NLLBConfig:
    max_chars: int = 512
    workers: int = 4                   # number of GPU instances
    per_worker_concurrency: int = 2    # streams/micro-batches per GPU
    alpha_ms: float = 6.0              # fixed overhead per NLLB chunk
    beta_ms_per_char: float = 0.030    # per-char time (measure!)
    
    


class TranslationUtils:
    
    
    @staticmethod
    def get_default_google_config() -> GoogleConfig:
        return GoogleConfig(max_chars=4800, alpha_ms=120, beta_ms_per_char=0.010)
    
    @staticmethod
    def get_defaultnllb_config() -> NLLBConfig:
        return NLLBConfig(max_chars=512, workers=4, per_worker_concurrency=2, alpha_ms=6, beta_ms_per_char=0.030)
    
    
    # ---------------------------
    # Service-time models
    # ---------------------------
    @staticmethod
    def google_service_time_ms(L: int, g: GoogleConfig) -> float:
        """
        Sequential Google model (single-thread): if L exceeds max_chars, we must do multiple back-to-back requests.
        """
        chunks = ceil(L / g.max_chars)
        return chunks * g.alpha_ms + L * g.beta_ms_per_char


    @staticmethod
    def nllb_service_time_ms(L: int, n: NLLBConfig) -> float:
        """
        NLLB model per article (processed on one slot): multiple 512-char chunks; we account for overhead per chunk.
        """
        chunks = ceil(L / n.max_chars)
        return chunks * n.alpha_ms + L * n.beta_ms_per_char

    # ---------------------------
    # Splitter with queue-aware routing
    # ---------------------------

    def split_feeds_for_translation_single_google(
            feeds: Iterable[T],
            text_getter: Callable[[T], str],
            gcfg: GoogleConfig,
            ncfg: NLLBConfig,
            hard_cutover_len: int | None = None,
            bias_google_ms: float = 0.0,
        ) -> Tuple[List[T], List[T]]:
        """
        Returns (feeds_nllb, feeds_google)
        - Google is single-threaded (web scraping) → 1 queue with 'g_backlog_ms'
        - NLLB has S = workers * per_worker_concurrency slots → min-heap of slot backlogs
        - Each item is routed to the side that minimizes (current_backlog + service_time)

        Parameters:
            hard_cutover_len: optional static rule; if L >= hard_cutover_len, prefer Google
                            (still overridden if Google backlog makes it slower).
            bias_google_ms:   optional penalty to Google (e.g., +200 ms) to further reduce Google usage.

        Complexity:
            O(N log S) where S is small (e.g., <= 16). Scales to tens/hundreds of thousands easily.
        """
        # Initialize NLLB slots as a min-heap of backlogs (ms)
        slots = ncfg.workers * ncfg.per_worker_concurrency
        if slots < 1:
            raise ValueError("NLLB must have at least 1 slot")

        nllb_heap = [0.0] * slots  # each entry = backlog_ms on that slot
        heapq.heapify(nllb_heap)

        g_backlog_ms = 0.0  # single Google queue backlog (ms)

        feeds_nllb: List[T] = []
        feeds_google: List[T] = []

        for feed in feeds:
            text = text_getter(feed)
            L = len(text)

            # Compute service times
            g_svc = TranslationUtils.google_service_time_ms(L, gcfg) + bias_google_ms
            n_svc = TranslationUtils.nllb_service_time_ms(L, ncfg)

            # Estimated finish times *if we add this item now*
            # Google: one queue
            g_finish = g_backlog_ms + g_svc

            # NLLB: assign to least-loaded slot (peek heap)
            least_slot_load = nllb_heap[0]
            n_finish = least_slot_load + n_svc

            # Optional static rule to nudge very long texts to Google
            if hard_cutover_len is not None and L >= hard_cutover_len:
                # Only send to Google if it's not worse than NLLB (protects against huge Google backlog)
                if g_finish <= n_finish:
                    feeds_google.append(feed)
                    g_backlog_ms = g_finish  # push into Google queue
                else:
                    # send to NLLB anyway (Google too backed up)
                    feeds_nllb.append(feed)
                    heapq.heapreplace(nllb_heap, least_slot_load + n_svc)
                continue

            # Dynamic routing (minimize finish time)
            if g_finish <= n_finish:
                feeds_google.append(feed)
                g_backlog_ms = g_finish
            else:
                feeds_nllb.append(feed)
                heapq.heapreplace(nllb_heap, least_slot_load + n_svc)

        return feeds_nllb, feeds_google
