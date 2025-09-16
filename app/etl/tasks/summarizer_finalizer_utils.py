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

import logging
import re
import json
import hashlib
from typing import Dict, Any, List, Optional

import torch
from transformers import GenerationConfig  # ensure this is imported


class SummarizerFinalizerUtils:
    logger = logging.getLogger(__name__)

    # -------------------- PUBLIC API --------------------
    @staticmethod
    def _summarizer_finalizer(
        text: str,
        model,            # AutoModelForSeq2SeqLM (google/flan-t5-base)
        tokenizer,        # matching AutoTokenizer
        device,           # torch.device
        max_tokens: int,  # desired OUTPUT budget for summary
    ) -> Dict[str, Any]:
        """
        Clean + summarize noisy news text with flan-t5-base and return news-centric metadata.
        Returns:
            {
              "summary": str,
              "meta": { ... news meta ... },
              "debug": { "input_tokens": int, "prompt_tokens": int, "output_tokens": int, "truncated": bool }
            }
        """
        if not text or not text.strip():
            return {"summary": "", "meta": {}, "debug": {"input_tokens": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated": False}}

        cleaned = SummarizerFinalizerUtils._preclean_text(text)
        prompt = SummarizerFinalizerUtils._build_instruction_prompt(cleaned)

        # Respect T5 input window (~512). Keep headroom.
        model_max = getattr(tokenizer, "model_max_length", 512)
        if model_max is None or model_max > 512:
            model_max = 512
        input_cap = min(480, model_max - 24)

        enc = tokenizer(prompt, max_length=input_cap, truncation=True, padding=True, return_tensors="pt").to(device)
        input_tokens = int(enc.input_ids.shape[1])
        # crude truncation flag
        truncated = tokenizer(prompt, truncation=True, max_length=input_cap, return_tensors="pt").input_ids.shape[1] < \
                    tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        # Deterministic generation
        gc = SummarizerFinalizerUtils._generation_config()
        #check this part
        safe_max_out = max(80, min(180, max_tokens if isinstance(max_tokens, int) else 160))
        safe_min_out = min(64, max(0, safe_max_out // 3))
        setattr(gc, "max_new_tokens", safe_max_out)
        setattr(gc, "min_new_tokens", safe_min_out)

        with torch.inference_mode():
            outputs = model.generate(**enc, generation_config=gc)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        summary = SummarizerFinalizerUtils._postclean_summary(summary)

        # Output token count (approx)
        out_ids = tokenizer(summary, return_tensors="pt", add_special_tokens=False).input_ids
        output_tokens = int(out_ids.shape[1])

        # ---------- NEWS META (LLM JSON + heuristics fallback) ----------
        news_meta = SummarizerFinalizerUtils._extract_news_meta(summary, model, tokenizer, device)

        # Add a stable signature for dedup/cluster keys
        news_meta["signature_sha1"] = hashlib.sha1(summary.lower().encode("utf-8")).hexdigest()
        news_meta["length_chars"] = len(summary)

        return {
            "summary": summary,
            "meta": news_meta,
            "debug": {
                "input_tokens": input_tokens,
                "prompt_tokens": input_tokens,  # prompt == encoded input
                "output_tokens": output_tokens,
                "truncated": bool(truncated),
            },
        }

    # -------------------- HELPERS --------------------

    @staticmethod
    def _generation_config() -> GenerationConfig:
        return GenerationConfig(
            num_beams=4,
            length_penalty=1.1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            do_sample=False,
        )

    @staticmethod
    def _build_instruction_prompt(text: str) -> str:
        return (
            "Task: Clean and summarize the following news text.\n"
            "- Remove ads, cookie notices, trackers, boilerplate, placeholders.\n"
            "- Keep only the main event facts.\n"
            "- Output concise sentences suitable for embedding and clustering.\n\n"
            f"Text:\n{text}"
        )

    @staticmethod
    def _preclean_text(text: str) -> str:
        patterns = [
            r"(?i)\bcookie(s)? (notice|policy)\b.*?$",
            r"(?i)\bad(s)?|advertisement|sponsored\b.*?$",
            r"(?i)\bprivacy (notice|policy)\b.*?$",
            r"(?i)\bsubscribe now\b.*?$",
            r"(?i)\b(login|sign in) to continue\b.*?$",
            r"\(\s*AP\s*\)",
            r"(?i)\bby\s*,?\s*by\s*,?\s*and\s*received\s*from\b.*?$",
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.MULTILINE)
        # de-dup lines + whitespace
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = "\n".join(dict.fromkeys(lines))
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _postclean_summary(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^[\"'–—-]\s*", "", s)
        # limit to max 4 sentences (simple heuristic)
        parts = re.split(r"(?<=[.!?])\s+", s)
        if len(parts) > 4:
            s = " ".join(parts[:4]).strip()
        return s

    # ---------- NEWS META EXTRACTION ----------
    @staticmethod
    def _extract_news_meta(summary: str, model, tokenizer, device) -> Dict[str, Any]:
        """
        Prefer LLM JSON extraction (flan-t5-base) with a strict JSON-only prompt,
        then fallback to heuristics if JSON parse fails.
        """
        prompt = (
            "Extract structured NEWS METADATA as strict JSON (no prose). "
            "Keys: domain (one of: politics, business, tech, culture, sports, science, health, environment, world, local), "
            "topics (list of short keywords)"
            "entities {people:[], orgs:[], places:[]}, event_date (ISO if known else null), "
            "geo {country:null|str, region:null|str, city:null|str}.\n"
            "Only output valid JSON.\n\n"
            f"Text:\n{summary}"
        )

        # Encode small prompt (summary is short already)
        enc = tokenizer(prompt, max_length=480, truncation=True, padding=True, return_tensors="pt").to(device)
        gc = GenerationConfig(num_beams=1, do_sample=False)
        setattr(gc, "max_new_tokens", 180)
        setattr(gc, "min_new_tokens", 40)

        try:
            with torch.inference_mode():
                out = model.generate(**enc, generation_config=gc)
            txt = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            meta = SummarizerFinalizerUtils._safe_json_loads(txt)
            if isinstance(meta, dict) and "domain" in meta and "topics" in meta:
                return SummarizerFinalizerUtils._normalize_meta(meta)
        except Exception:
            pass

        # Heuristic fallback
        return SummarizerFinalizerUtils._heuristic_meta(summary)

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        # remove possible code fences or trailing junk
        s = s.strip()
        s = re.sub(r"^```(json)?", "", s)
        s = re.sub(r"```$", "", s).strip()
        # try direct
        try:
            return json.loads(s)
        except Exception:
            # attempt to find the first {...} block
            m = re.search(r"\{.*\}", s, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {}

    @staticmethod
    def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        # ensure shapes/types and defaults
        domain = (meta.get("domain") or "world").lower()
        topics = meta.get("topics") or []
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(",") if t.strip()]
        entities = meta.get("entities") or {}
        people = entities.get("people") or []
        orgs = entities.get("orgs") or []
        places = entities.get("places") or []
        geo = meta.get("geo") or {}
        norm = {
            "domain": domain,
            "topics": topics[:6],
            "is_politics_central": bool(meta.get("is_politics_central", False)),
            "entities": {
                "people": list(dict.fromkeys([str(x).strip() for x in people]))[:10],
                "orgs":   list(dict.fromkeys([str(x).strip() for x in orgs]))[:10],
                "places": list(dict.fromkeys([str(x).strip() for x in places]))[:10],
            },
            "event_date": meta.get("event_date") or None,
            "geo": {
                "country": geo.get("country") or None,
                "region":  geo.get("region") or None,
                "city":    geo.get("city") or None,
            },
        }
        return norm

    @staticmethod
    def _heuristic_meta(summary: str) -> Dict[str, Any]:
        # Very lightweight keyword tagging as a fallback
        text = summary.lower()

        domain_map = {
            "politics": ["president", "election", "parliament", "minister", "congress", "bolsonaro", "trump", "moraes"],
            "business": ["market", "startup", "acquisition", "merger", "ipo", "revenue", "profit"],
            "tech":     ["ai", "software", "chip", "semiconductor", "cyber", "app", "cloud"],
            "culture":  ["exhibit", "museum", "festival", "art", "fashion", "show"],
            "sports":   ["match", "cup", "league", "tournament", "goal"],
            "science":  ["research", "study", "scientist", "lab", "space"],
            "health":   ["hospital", "vaccine", "virus", "health", "disease"],
            "environment": ["amazon", "climate", "emissions", "deforestation", "wildfire"],
            "world":    [ "conflict", "border", "diplomatic"],
            "local":    ["mayor", "city council", "municipal"],
        }
        domain_scores = {k: 0 for k in domain_map}
        for k, kws in domain_map.items():
            for w in kws:
                if w in text:
                    domain_scores[k] += 1
        domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else "world"

        is_politics_central = domain == "politics"

        # crude entity pulls
        people = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", summary)  # First Last
        orgs = re.findall(r"\b([A-Z]{2,}(?: [A-Z]{2,})*)\b", summary)    # ALLCAPS blocks
        places = re.findall(r"\b([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?)\b", summary)

        topics = list(dict.fromkeys(re.findall(r"\b[a-z]{4,}\b", text)))[:6]

        return {
            "domain": domain,
            "topics": topics,
            "is_politics_central": is_politics_central,
            "entities": {
                "people": list(dict.fromkeys(people))[:10],
                "orgs":   list(dict.fromkeys(orgs))[:10],
                "places": list(dict.fromkeys(places))[:10],
            },
            "event_date": None,
            "geo": {"country": None, "region": None, "city": None},
        }