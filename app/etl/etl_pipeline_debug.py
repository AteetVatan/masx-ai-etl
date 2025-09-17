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


class ETLPipelineDebug:
    """ETLPipelineDebug class for loading and storing summarized feeds from JSON file for the given flashpoint."""
    
    @staticmethod
    def load_summarized_feeds(flashpoint_id: str, date: str):
        """Load summarized feeds from JSON file for the given flashpoint."""
        from app.etl_data.etl_models.feed_model import FeedModel
        import json
        from pathlib import Path
        from datetime import datetime

        # Create flashpoint-specific file path
        path = Path(f"debug_data/summarized_feeds_{flashpoint_id}_{date}.json")

        # Best-first: UTF-8; fallback to UTF-8 with BOM; final fallback replaces bad bytes
        def load_json_textsafe(path: Path):
            for enc in ("utf-8", "utf-8-sig"):
                try:
                    with path.open("r", encoding=enc) as f:
                        return json.load(f)
                except UnicodeDecodeError:
                    continue
            # last resort: don't crash; replace undecodable bytes
            with path.open("r", encoding="utf-8", errors="replace") as f:
                return json.load(f)

        def convert_datetime_strings(feed_data: dict) -> dict:
            """Convert ISO datetime strings back to datetime objects."""
            for field in ['created_at', 'updated_at']:
                if field in feed_data and feed_data[field] and isinstance(feed_data[field], str):
                    try:
                        feed_data[field] = datetime.fromisoformat(feed_data[field])
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string or set to None
                        feed_data[field] = None
            return feed_data

        if not path.exists():           
            return []

        summarized_feeds_json = load_json_textsafe(path)
        
        # Convert datetime strings back to datetime objects
        for feed_data in summarized_feeds_json:
            convert_datetime_strings(feed_data)
        
        summarized_feeds = [FeedModel(**feed) for feed in summarized_feeds_json]
        return summarized_feeds

    @staticmethod
    def _store_summarized_feeds(summarized_feeds: list, flashpoint_id: str, date: str):
        """Store summarized feeds as JSON file for the given flashpoint."""
        import json
        import re
        from pathlib import Path
        from datetime import datetime

        def clean_text(val):
            if isinstance(val, str):
                # Remove newlines and tabs
                val = val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                # Collapse multiple spaces into one
                val = re.sub(r"\s+", " ", val).strip()
                return val
            elif isinstance(val, list):
                return [clean_text(v) for v in val]
            elif isinstance(val, dict):
                return {k: clean_text(v) for k, v in val.items()}
            elif isinstance(val, datetime):
                # Convert datetime to ISO format string
                return val.isoformat()
            return val

        # Create debug_data directory if it doesn't exist
        debug_dir = Path("debug_data")
        debug_dir.mkdir(exist_ok=True)
        
        # Create flashpoint-specific file path
        file_path = debug_dir / f"summarized_feeds_{flashpoint_id}_{date}.json"
        
        # Clean and serialize the feeds
        cleaned_feeds = [clean_text(feed.dict()) for feed in summarized_feeds]
        json_str = json.dumps(cleaned_feeds, ensure_ascii=False, indent=4)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        return str(file_path)
