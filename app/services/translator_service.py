from app.config import get_service_logger
from deep_translator import GoogleTranslator


class TranslatorService:

    _google_translator: GoogleTranslator | None = None
    _logger = get_service_logger("Translator")

    @classmethod
    def get_google_translator(cls, lang="en", proxies=None) -> GoogleTranslator:
        if not cls._google_translator or proxies is not None:
            cls._google_translator = None
            cls.__load_google_translator(lang, proxies)

        return cls._google_translator

    @classmethod
    def __load_google_translator(cls, lang, proxies=None):
        try:
            if proxies is not None:
                cls._google_translator = GoogleTranslator(
                    source="auto", target=lang, proxies=proxies
                )
            else:
                cls._google_translator = GoogleTranslator(source="auto", target=lang)
        except Exception as e:
            cls._logger.error(f"model_manager.py:Failed to load GoogleTranslator: {e}")
            raise RuntimeError(f"Failed to load GoogleTranslator: {e}")
