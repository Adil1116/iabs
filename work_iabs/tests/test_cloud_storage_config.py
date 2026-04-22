from __future__ import annotations

from pathlib import Path

from brain.config import AppConfig


def test_from_env_supports_dotenv_and_supabase_alias(monkeypatch, tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text(
        '\n'.join(
            [
                'SUPABASE_DB_URL=postgresql://user:secret@db.supabase.co:6543/postgres?sslmode=require',
                'IABS_MEMORY_PATH=',
                f'IABS_DATA_DIR={tmp_path / "data"}',
            ]
        ),
        encoding='utf-8',
    )

    for key in ('IABS_MEMORY_DSN', 'DATABASE_URL', 'SUPABASE_DB_URL', 'SUPABASE_DATABASE_URL', 'IABS_MEMORY_PATH', 'IABS_DATA_DIR'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('IABS_ENV_FILE', str(env_file))

    config = AppConfig.from_env()

    assert config.memory_dsn == 'postgresql://user:secret@db.supabase.co:6543/postgres?sslmode=require'
    assert config.memory_path is None


def test_from_env_falls_back_to_sqlite_when_dsn_missing(monkeypatch, tmp_path):
    env_file = tmp_path / '.env'
    env_file.write_text(
        f'IABS_DATA_DIR={tmp_path / "appdata"}\nIABS_MEMORY_PATH=\n',
        encoding='utf-8',
    )

    for key in ('IABS_MEMORY_DSN', 'DATABASE_URL', 'SUPABASE_DB_URL', 'SUPABASE_DATABASE_URL', 'IABS_MEMORY_PATH', 'IABS_DATA_DIR'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('IABS_ENV_FILE', str(env_file))

    config = AppConfig.from_env()

    assert config.memory_dsn is None
    assert config.memory_path == Path(tmp_path / 'appdata' / 'memory_store.sqlite').resolve()
