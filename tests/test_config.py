from __future__ import annotations

from app.config import Settings, get_settings


def test_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("DB_TYPE", "postgres")
    monkeypatch.setenv("DB_HOST", "db.local")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "analytics")
    monkeypatch.setenv("DB_USER", "analyst")
    monkeypatch.setenv("DB_PASSWORD", "secret")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.normalized_ollama_model == "mistral:latest"
    assert settings.db_port == 5433
    assert settings.database_target == "postgres://analyst@db.local:5433/analytics"
    assert settings.sqlalchemy_url.startswith("postgresql+psycopg2://analyst:secret@db.local:5433/")

    get_settings.cache_clear()


def test_vanna_ui_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("VANNA_UI_TITLE", "My Vanna")
    monkeypatch.setenv("VANNA_UI_SUBTITLE", "My Subtitle")
    monkeypatch.setenv("VANNA_UI_CDN_URL", "https://cdn.example.com/vanna.js")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.vanna_ui_title == "My Vanna"
    assert settings.vanna_ui_subtitle == "My Subtitle"
    assert settings.vanna_ui_cdn_url == "https://cdn.example.com/vanna.js"

    get_settings.cache_clear()


def test_validate_database_config_raises_for_missing_values(tmp_path):
    settings = Settings(
        _env_file=None,
        db_name="",
        db_user="",
        db_password="",
        chroma_path=tmp_path / "chroma",
        training_data_dir=tmp_path / "training",
        business_glossary_path=tmp_path / "training" / "glossary.md",
        example_pairs_path=tmp_path / "training" / "pairs.json",
        bootstrap_state_path=tmp_path / "training" / "bootstrap_state.json",
    )

    try:
        settings.validate_database_config()
    except ValueError as exc:
        assert "DB_NAME" in str(exc)
        assert "DB_USER" in str(exc)
        assert "DB_PASSWORD" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected validate_database_config() to raise.")
