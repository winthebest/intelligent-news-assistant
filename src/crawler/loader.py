import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)


def load_articles(file_path: Optional[Path] = None) -> List[Dict]:
    """Load articles from JSON; defaults to ``RAW_DATA_DIR/vnexpress_articles.json``."""
    if file_path is None:
        file_path = settings.RAW_DATA_DIR / "vnexpress_articles.json"
    
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading articles from {file_path}: {e}")
        return []


def load_documents(file_path: Optional[Path] = None) -> List[Dict]:
    """Load processed documents (chunks); defaults to ``PROCESSED_DATA_DIR/documents.json``."""
    if file_path is None:
        file_path = settings.PROCESSED_DATA_DIR / "documents.json"

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading documents from {file_path}: {e}")
        return []
