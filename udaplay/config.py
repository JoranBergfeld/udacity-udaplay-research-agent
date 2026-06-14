"""Centralized configuration and client factories.

All OpenAI access goes through the SDK, which auto-reads OPENAI_API_KEY and
OPENAI_BASE_URL from the environment (Vocareum proxy). ChromaDB's embedding
function does NOT read OPENAI_BASE_URL, so it is wired separately in
index_builder.py using OPENAI_BASE_URL below.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Models
CHAT_MODEL = os.getenv("UDAPLAY_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("UDAPLAY_EMBED_MODEL", "text-embedding-ada-002")

# Endpoints / keys
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. https://openai.vocareum.com/v1
CHROMA_OPENAI_KEY_ENV = "CHROMA_OPENAI_API_KEY"

# Paths
CHROMA_PATH = os.getenv("UDAPLAY_CHROMA_PATH", "chromadb")
DATA_DIR = os.getenv("UDAPLAY_DATA_DIR", os.path.join("data", "games"))
CACHE_DIR = os.getenv("UDAPLAY_CACHE_DIR", os.path.join(".cache", "rawg"))


def get_openai_client():
    """Return an OpenAI client that respects OPENAI_BASE_URL / OPENAI_API_KEY from env."""
    from openai import OpenAI

    return OpenAI()
