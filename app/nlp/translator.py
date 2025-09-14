from transformers import pipeline
import logging


class Translator:
    logger = logging.getLogger(__name__)
    
    @staticmethod
    def translate(text: str, src_lang: str, tgt_lang: str, model, tokenizer, device, max_tokens) -> str:
        
        try:
            pipeline = Translator.get_pipeline(src_lang, tgt_lang, model, tokenizer, device, max_tokens)
            result = pipeline(text)
            return result[0]["translation_text"]
        except Exception as e:
            Translator.logger.error(f"translator.py:Translation failed: {e}")
            raise e
            #return text
        
    @staticmethod
    def get_pipeline(src_lang: str, tgt_lang: str, model, tokenizer, device, max_tokens):
        """
        Returns a cached pipeline for the given src→tgt translation.
        """      
           
            #from app.core.concurrency.device import get_torch_device

            #device = get_torch_device()
        try:
            hf_pipeline = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=max_tokens,
                device=device,
            )
            return hf_pipeline
        except Exception as e:
            Translator.logger.error(f"translator.py:get_pipeline failed: {e}")
            return None