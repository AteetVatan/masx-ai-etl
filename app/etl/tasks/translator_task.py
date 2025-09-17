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


from app.config import get_service_logger, get_settings
#from app.etl.tasks import Translator
from app.etl_data.etl_models import FeedModel
from app.core.exceptions import ServiceException
from app.core.concurrency import InferenceRuntime, RuntimeConfig
from app.services import TranslatorService
from typing import Optional
#from app.enumeration.enums import TaskEnums
from app.nlp import Translator, NLPUtils
from app.core.model import TranslatorModelManager
from app.core.concurrency import CPUExecutors
from app.enumeration import WorkloadEnums
from app.constants import ISO_TO_NLLB_MERGED
from app.services import ProxyService


import re
import asyncio
import time


class TranslatorTask:

    def __init__(self):
        self.logger = get_service_logger("TranslatorTask")
        self.translator = Translator()
        self.settings = get_settings()
        # Initialize inference runtime
        self.inference_runtime: Optional[InferenceRuntime] = None
        self._proxy_service = ProxyService()      
        self.cpu_executors = CPUExecutors(workload=WorkloadEnums.CPU)
        self.translator_service = TranslatorService()
        
        

    async def translate_all_feeds(self, feeds: list[FeedModel]) -> list[FeedModel]:
        try:
            
            # get feeds which are not in english
            feeds_multilingual = await self._get_feeds_multilingual(feeds)
            self.logger.info(f"TranslatorTask:translate_all_feeds {len(feeds_multilingual)} feeds")
            if len(feeds_multilingual) == 0:
                return feeds
            
            
            # Initialize inference runtime if not already done
            if not self.inference_runtime:
                self.logger.info(
                    f"TranslatorTask:Initializing inference runtime"
                )
                #await self.concurrency_utils.initialize_inference_runtime()
                await self._initialize_inference_runtime()            
           
            # seperate the feeds with language in NLLB Model Supported Languages (ISO_TO_NLLB_MERGED)
            # other wise use google translate
            #feeds_nllb = [feed for feed in feeds_multilingual if feed.language in ISO_TO_NLLB_MERGED]
            #feeds_google = [feed for feed in feeds_multilingual if feed.language not in ISO_TO_NLLB_MERGED]

            
            # Shorts → GPU (NLLB).
            # Longs → Google Translate.           
            
            #feeds_nllb, feeds_google = await self.divide_feeds_nllb_google(feeds_multilingual)
            #for now use all feeds for nllb
            # only error fallbacks to google translate
            
            feeds_nllb = []
            feeds_google = []
            
            if self.settings.debug:
                feeds_nllb = feeds_nllb[:1]
                feeds_google = feeds_google[1:]
            else:
                #by default use all feeds for nllb
                feeds_nllb = feeds_multilingual            
            

            result_nllb, result_google = await asyncio.gather(
                self.translate_all_feeds_nllb(feeds_nllb),
                self.translate_all_feeds_google(feeds_google)
            )
            result = result_nllb + result_google
            
            #final merge
            for feed in feeds:
                feed_item = next((r_feed for r_feed in result if feed.id == r_feed.id), None)
                if feed_item:
                    feed.raw_text_en = feed_item.raw_text_en
                    feed.processed_text = feed_item.processed_text           
            
            return feeds

        except Exception as e:
            self.logger.error(f"TranslatorTask:Error translating feeds: {e}")
            raise ServiceException(f"Error translating feeds: {e}")
        finally:
            if self.inference_runtime:
                self.inference_runtime.model_manager.cleanup()
            if self.cpu_executors:
                self.cpu_executors.shutdown(wait=True)



    async def translate_all_feeds_nllb(self, feeds: list[FeedModel]) -> list[FeedModel]:
        try:
            if len(feeds) == 0:
                return []
            
            self.logger.info(
                f"TranslatorTask:translate_all_feeds_nllb {len(feeds)} feeds"
            )

            # # Initialize inference runtime if not already done
            # if not self.inference_runtime:
            #     self.logger.info(
            #         f"TranslatorTask:Initializing inference runtime"
            #     )
            #     #await self.concurrency_utils.initialize_inference_runtime()
            #     await self._initialize_inference_runtime()

        
            translated_feeds: list[FeedModel] = []
            # chucked batched according to model max tokens
            chuncks_with_id = self._create_batches_with_chunks(feeds)
            translated_feeds = await self._process_batch(chuncks_with_id, feeds)
            return translated_feeds
        except Exception as e:
            self.logger.error(f"TranslatorTask:Error translating feeds: {e}")
            raise ServiceException(f"Error translating feeds: {e}")
       
       
    async def _get_feeds_multilingual(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Return only feeds whose raw_text is not detected as English.
        Uses thread offloading for language detection.
        """
        return [feed for feed in feeds if feed.language != "en"]



    async def _process_batch(self, chuncks_with_id, feeds):
        """Process a batch of feeds using the inference runtime with CPI/GPU micro-batching."""       
        try:           
            translator_manager: TranslatorModelManager = self.inference_runtime.model_manager
            batch_size = translator_manager.pool_size

            async def run(item: tuple[int, str, str, str]) -> tuple[int, str, str, str] | None:
                # each task acquires its own instance
                async with translator_manager.acquire(destroy_after_use=False) as instance:
                    try:
                        feed_index, feed_id, language, chunk = item
                        source_lang = language
                        target_lang = ISO_TO_NLLB_MERGED["en"]
                        #result = Translator.translate(chunk, source_lang, target_lang, instance.model, instance.tokenizer, instance.device, instance.max_tokens)           
            
                        result = await self.cpu_executors.run_in_thread(
                             Translator.translate,  chunk, source_lang, target_lang, instance.model, instance.tokenizer, instance.device, instance.max_tokens
                        )
                        if result is None:
                            raise Exception("Translator: Error translating feed")                        
                        
                        return (feed_index, feed_id, source_lang, result)
                    except Exception as e:
                        #fallback to google translate
                        try:
                            google_translator = self.translator_service.get_google_translator(lang="en", proxies=None)
                            result = google_translator.translate(chunk)
                            return (feed_index, feed_id, source_lang, result)
                        except Exception as e:
                            self.logger.error(f"Translator: Error translating feed: {e}")
                            return (feed_index, feed_id, source_lang, "")



            #(feed_index_dict[feed_id], feed_id, language_dict[feed_id], chunk)
            #group the chuncks_with_id by batches (batch_size is the number of model instances)
            batches = [chuncks_with_id[i:i+batch_size] for i in range(0, len(chuncks_with_id), batch_size)]

            results = []
            #time_start = time.time()
            for batch in batches:
                tasks = [run(chunk_item) for chunk_item in batch]
                results.extend(await asyncio.gather(*tasks, return_exceptions=True))
            #time_end = time.time()
            #self.logger.info(f"TranslatorTask:Time taken : {time_end - time_start} seconds for {len(feeds)} feeds ")
            
            #order them by feed_id and feed_index
            results.sort(key=lambda x: (x[1], x[0]))
            
            #combine the results by feed_id and feed_index and merge the chunks
            results_dict = {}
            for r in results:
                if r is None:
                    continue
                if isinstance(r, Exception):
                    self.logger.error(f"Translator: task exception: {r}")
                    continue
                feed_index, feed_id, source_lang, chunk = r
                if feed_id not in results_dict:
                    results_dict[feed_id] = []
                results_dict[feed_id].append(chunk)
                    
            for feed_id, chunks in results_dict.items():
                results_dict[feed_id] = "\n\n".join(chunks)
             
             
            for feed in feeds:
              feed.raw_text_en = results_dict[feed.id]               
              feed.processed_text = results_dict[feed.id]              
           
            return feeds

        except Exception as e:
            self.logger.error(f"Summarizer: Batch processing failed: {e}")
            return []
        


    async def _initialize_inference_runtime(self):
        """Initialize the inference runtime for translator."""
        try:
            # Create runtime config optimized for summarization
            config = RuntimeConfig()


            # Create and start inference runtime
            self.inference_runtime = InferenceRuntime(
                model_manager_loader=self._get_translator_model_manager, config=config
            )

            await self.inference_runtime.start()
            self.logger.info(
                "translator.py:Translator:Inference runtime initialized for translator"
            )

        except Exception as e:
            self.logger.error(
                f"translator.py:Translator:Failed to initialize inference runtime: {e}"
            )
            raise

    def _get_translator_model_manager(self):
        """Model loader function for the inference runtime."""
        # Return only the model, not the tuple, since GPUWorker expects a single model
        return TranslatorModelManager(self.settings)

    def _create_batches_with_chunks(self, feeds: list[FeedModel]) -> list[tuple[int, str, str, str]]:
        """Create batches with chunks for the translator.
            # Split all chunks across batch_size groups.
            # Process each group in parallel (one per model instance).
            # Recombine processed chunks back under their original id.
            
            (feed_index , feed_id, language, chunk)
        """
        try:
            chunck_dict = {}
            language_dict = {}
            max_tokens = self.inference_runtime.model_manager.max_tokens         
            for feed in feeds:
                chunks = NLPUtils.split_text_smart(feed.processed_text, max_tokens-100)
                chunck_dict[feed.id] = chunks
                language_dict[feed.id] = ISO_TO_NLLB_MERGED[feed.language] if feed.language in ISO_TO_NLLB_MERGED else feed.language

            #Flatten all chunks with IDs
            chuncks_with_id = []
            feed_index_dict = {}
            for feed_id, chunks in chunck_dict.items():
                feed_index_dict[feed_id] = 0
                for chunk in chunks:
                    chuncks_with_id.append((feed_index_dict[feed_id], feed_id, language_dict[feed_id], chunk))
                    feed_index_dict[feed_id] += 1

            #Split into groups equal to number of model instances
            #batches = [chunck_with_id[i::batch_size] for i in range(batch_size)]

            return chuncks_with_id
        except Exception as e:
            self.logger.error(f"TranslatorTask:Error creating batches with chunks: {e}")
            return {}

    async def translate_all_feeds_google(self, feeds: list[FeedModel]) -> list[FeedModel]:
        """
        Translate the text to English using Google Translate.
        For now sync need a better solution.
        """
        try:
            if len(feeds) == 0:
                return []
            
            
            self.logger.info(
                f"TranslatorTask:translate_all_feeds_google {len(feeds)} feeds"
            )

            for feed in feeds:
                feed.processed_text = await self.google_translate_to_english(feed.processed_text)
                feed.raw_text_en = feed.processed_text

            return feeds

        except Exception as e:
            self.logger.error(f"TranslatorTask:Error translating feeds_google: {e}")
            raise ServiceException(f"Error translating feeds_google: {e}")
        finally:
            pass

    async def google_translate_to_english(self, text: str) -> str:
        """
        Translate the text to English.
        """
        try:
            
            #proxies = await self._proxy_service.get_proxies()
            google_translator = self.translator_service.get_google_translator(lang="en", proxies=None)
            
            # split the text into chunks, as the google translator has a limit of 5000 characters?
            self.logger.info(f"translator.py:[Translation] Using Google Translate")
            chunks = NLPUtils.split_text_smart(text, 4000) # google limit is 5000 characters
            translated_chunks = [
                self._google_safe_translate(google_translator,NLPUtils.clean_text(chunk)) for chunk in chunks
            ]
            return "\n\n".join(translated_chunks)
        except Exception as e:
            self.logger.error(f"translator.py:[CRITICAL] Full translation failed: {e}")
            return text


    def _google_safe_translate(self, google_translator, chunk: str) -> str:
        retries = 3
        delay = 2
        for attempt in range(retries):
            try:
                return google_translator.translate(chunk)
            except Exception as e:
                print(f"[Attempt {attempt + 1}/{retries}] Translation failed: {e}")
                time.sleep(delay)
        print(f"[Fallback] Using original chunk:\n{chunk[:80]}...")
        return chunk
    
    
    async def divide_feeds_nllb_google(self, feeds: list[FeedModel]) -> tuple[list[FeedModel], list[FeedModel]]:
        """
        Divide the feeds into nllb and google feeds.
        """
        try:
            if len(feeds) == 0:
                return [], []
            
            from app.etl.tasks import TranslationUtils
            
            gcfg = TranslationUtils.get_default_google_config()
            ncfg = TranslationUtils.get_defaultnllb_config()
            
            #for nllb
            translator_manager: TranslatorModelManager = self.inference_runtime.model_manager
            gpu_pool_size = translator_manager.pool_size
            ncfg.workers = gpu_pool_size
            ncfg.per_worker_concurrency = 1

            def get_processed_text(feed: FeedModel) -> str:
                return feed.processed_text
               
            feeds_nllb, feeds_google = TranslationUtils.split_feeds_for_translation_single_google(
                feeds, get_processed_text, gcfg, ncfg
            )     
      
            return feeds_nllb, feeds_google
        
        except Exception as e:
            self.logger.error(f"TranslatorTask:Error dividing feeds_nllb_google: {e}")
            raise ServiceException(f"Error dividing feeds_nllb_google: {e}")