import importlib

import udaplay.config as config


def test_defaults_present():
    assert config.CHAT_MODEL == "gpt-4o-mini"
    assert config.EMBED_MODEL == "text-embedding-ada-002"
    assert config.CHROMA_PATH
    assert config.DATA_DIR.endswith("games")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("UDAPLAY_CHAT_MODEL", "gpt-4o")
    importlib.reload(config)
    assert config.CHAT_MODEL == "gpt-4o"
    monkeypatch.delenv("UDAPLAY_CHAT_MODEL")
    importlib.reload(config)
