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

# merge ISO_TO_NLLB and ISO_TO_NLLB_1

ISO_TO_NLLB = {
    "ace": "ace_Latn",  # Acehnese (Latn)
    "am": "amh_Ethi",  # Amharic
    "ar": "arb_Arab",  # Arabic
    "as": "asm_Beng",  # Assamese
    "az": "azj_Latn",  # North Azerbaijani
    "be": "bel_Cyrl",  # Belarusian
    "bem": "bem_Latn",  # Bemba
    "ber": "ber_Latn",  # Central Moroccan Amazigh (Latin) – if available
    "bg": "bul_Cyrl",  # Bulgarian
    "bn": "ben_Beng",  # Bengali
    "br": "bre_Latn",  # Breton
    "bs": "bos_Latn",  # Bosnian
    "ca": "cat_Latn",  # Catalan
    "ceb": "ceb_Latn",  # Cebuano
    "cs": "ces_Latn",  # Czech
    "cy": "cym_Latn",  # Welsh
    "da": "dan_Latn",  # Danish
    "de": "deu_Latn",  # German
    "dv": "div_Latn",  # Divehi; check availability
    "el": "ell_Grek",  # Greek
    "en": "eng_Latn",  # English
    "eo": "epo_Latn",  # Esperanto
    "es": "spa_Latn",  # Spanish
    "et": "est_Latn",  # Estonian
    "eus": "eus_Latn",  # Basque
    "fa": "pes_Arab",  # Persian (Farsi)
    "fi": "fin_Latn",  # Finnish
    "fr": "fra_Latn",  # French
    "fy": "fuv_Latn",  # Nigerian Fulfulde (check)
    "ga": "gle_Latn",  # Irish
    "gl": "glg_Latn",  # Galician
    "gu": "guj_Gujr",  # Gujarati
    "ha": "hau_Latn",  # Hausa
    "he": "heb_Hebr",  # Hebrew
    "hi": "hin_Deva",  # Hindi
    "hr": "hrv_Latn",  # Croatian
    "hu": "hun_Latn",  # Hungarian
    "hy": "hye_Armn",  # Armenian
    "id": "ind_Latn",  # Indonesian
    "ig": "ibo_Latn",  # Igbo
    "ilo": "ilo_Latn",  # Ilocano
    "is": "isl_Latn",  # Icelandic
    "it": "ita_Latn",  # Italian
    "ja": "jpn_Jpan",  # Japanese
    "jv": "jav_Latn",  # Javanese
    "ka": "kat_Geor",  # Georgian
    "kk": "kaz_Cyrl",  # Kazakh
    "km": "khm_Khmr",  # Khmer
    "kn": "kan_Knda",  # Kannada
    "ko": "kor_Hang",  # Korean
    "ku": "kmr_Latn",  # Northern Kurdish
    "ky": "kir_Cyrl",  # Kyrgyz
    "la": "lat_Latn",  # Latin (if available)
    "lo": "lao_Laoo",  # Lao
    "lt": "lit_Latn",  # Lithuanian
    "lv": "lvs_Latn",  # Standard Latvian
    "mg": "plt_Latn",  # Plateau Malagasy (or "plt_Latn")
    "mi": "mri_Latn",  # Maori
    "mk": "mkd_Cyrl",  # Macedonian
    "ml": "mal_Mlym",  # Malayalam
    "mn": "mon_Cyrl",  # Mongolian (Halh) – check keyed code "khk_Cyrl"
    "mr": "mar_Deva",  # Marathi
    "ms": "msa_Latn",  # Malay (standard)
    "my": "mya_Mymr",  # Burmese
    "nb": "nob_Latn",  # Norwegian Bokmål
    "nd": "nso_Latn",  # Northern Sotho
    "nl": "nld_Latn",  # Dutch
    "nn": "nno_Latn",  # Norwegian Nynorsk
    "no": "nob_Latn",  # Norwegian (alias)
    "pl": "pol_Latn",  # Polish
    "ps": "pbt_Arab",  # Southern Pashto
    "pt": "por_Latn",  # Portuguese
    "ro": "ron_Latn",  # Romanian
    "ru": "rus_Cyrl",  # Russian
    "sd": "snd_Arab",  # Sindhi
    "si": "sin_Sinh",  # Sinhala
    "sk": "slk_Latn",  # Slovak
    "sl": "slv_Latn",  # Slovenian
    "so": "som_Latn",  # Somali
    "sq": "als_Latn",  # Tosk Albanian
    "sr": "srp_Cyrl",  # Serbian
    "st": "sot_Latn",  # Southern Sotho
    "sv": "swe_Latn",  # Swedish
    "sw": "swh_Latn",  # Swahili
    "ta": "tam_Taml",  # Tamil
    "te": "tel_Telu",  # Telugu
    "tg": "tgk_Cyrl",  # Tajik
    "th": "tha_Thai",  # Thai
    "tk": "tuk_Latn",  # Turkmen
    "tl": "tgl_Latn",  # Tagalog
    "tr": "tur_Latn",  # Turkish
    "tt": "tat_Cyrl",  # Tatar
    "ug": "uig_Arab",  # Uyghur
    "uk": "ukr_Latn",  # Ukrainian (often 'ukr_Cyrl', but NLLB uses Latn)
    "ur": "urd_Arab",  # Urdu
    "uz": "uzb_Latn",  # Uzbek (Latin)
    "vi": "vie_Latn",  # Vietnamese
    "xh": "xho_Latn",  # Xhosa
    "yi": "ydd_Latn",  # Yiddish – check if available
    "yo": "yor_Latn",  # Yoruba
    "zh": "zho_Hans",  # Chinese Simplified
    "zu": "zul_Latn",  # Zulu
}


ISO_TO_NLLB_1 = {
    "af": "afr_Latn",
    "ak": "aka_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "ay": "ayr_Latn",
    "az": "azj_Latn",
    "bm": "bam_Latn",
    "be": "bel_Cyrl",
    "bn": "ben_Beng",
    "bho": "bho_Deva",
    "bs": "bos_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "ceb": "ceb_Latn",
    "cs": "ces_Latn",
    "ckb": "ckb_Arab",
    "tt": "crh_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "ee": "ewe_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "gd": "gla_Latn",
    "ga": "gle_Latn",
    "gl": "glg_Latn",
    "gn": "grn_Latn",
    "gu": "guj_Gujr",
    "ht": "hat_Latn",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "nl": "nld_Latn",
    "ig": "ibo_Latn",
    "ilo": "ilo_Latn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "jv": "jav_Latn",
    "ja": "jpn_Jpan",
    "kab": "kab_Latn",
    "kn": "kan_Knda",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "rw": "kin_Latn",
    "ko": "kor_Hang",
    "ku": "kmr_Latn",
    "lo": "lao_Laoo",
    "lv": "lvs_Latn",
    "ln": "lin_Latn",
    "lt": "lit_Latn",
    "lb": "ltz_Latn",
    "lg": "lug_Latn",
    "lus": "lus_Latn",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "mk": "mkd_Cyrl",
    "mg": "plt_Latn",
    "mt": "mlt_Latn",
    "mni-Mtei": "mni_Beng",
    "mni": "mni_Beng",
    "mn": "khk_Cyrl",
    "mi": "mri_Latn",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "no": "nno_Latn",
    "ne": "npi_Deva",
    "ny": "nya_Latn",
    "om": "gaz_Latn",
    "or": "ory_Orya",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ps": "pbt_Arab",
    "qu": "quy_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sm": "smo_Latn",
    "sn": "sna_Latn",
    "sd": "snd_Arab",
    "so": "som_Latn",
    "es": "spa_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "tl": "tgl_Latn",
    "th": "tha_Thai",
    "ti": "tir_Ethi",
    "ts": "tso_Latn",
    "tk": "tuk_Latn",
    "tr": "tur_Latn",
    "ug": "uig_Arab",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "xh": "xho_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh-CN": "zho_Hans",
    "zh": "zho_Hans",
    "zh-TW": "zho_Hant",
    "zu": "zul_Latn",
    "pa": "pan_Guru",
}

# Merge with ISO_TO_NLLB_1 taking precedence
ISO_TO_NLLB_MERGED = {**ISO_TO_NLLB, **ISO_TO_NLLB_1}
ISO_TO_NLLB_MERGED = dict(sorted(ISO_TO_NLLB_MERGED.items()))
