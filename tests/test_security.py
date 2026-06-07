"""Principal + resolve_principal 单元测试。"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app.security import Principal, resolve_principal
from config import settings


# ---------- Principal.namespace_chat_id ----------


def test_namespace_chat_id_combines_principal_and_raw():
    p = Principal(principal_id="alice")
    assert p.namespace_chat_id("foo") == "alice:foo"


def test_namespace_chat_id_falls_back_to_default_for_empty():
    p = Principal(principal_id="alice")
    assert p.namespace_chat_id("") == "alice:default"
    assert p.namespace_chat_id(None) == "alice:default"


def test_principal_is_frozen():
    p = Principal(principal_id="alice")
    with pytest.raises(Exception):
        p.principal_id = "bob"  # type: ignore[misc]


def test_principal_is_anonymous_default_false():
    p = Principal(principal_id="alice")
    assert p.is_anonymous is False


# ---------- resolve_principal ----------


@pytest.fixture
def temp_security(monkeypatch):
    """允许测试临时改 settings.security 而不污染其他用例。"""
    original_keys = dict(settings.security.api_keys)
    original_allow = settings.security.allow_anonymous
    yield
    monkeypatch.setattr(settings.security, "api_keys", original_keys)
    monkeypatch.setattr(settings.security, "allow_anonymous", original_allow)


def _resolve(x_api_key):
    return asyncio.run(resolve_principal(x_api_key=x_api_key))


def test_resolve_with_valid_key(temp_security, monkeypatch):
    monkeypatch.setattr(settings.security, "api_keys", {"sk-alice": "alice"})
    p = _resolve("sk-alice")
    assert p.principal_id == "alice"
    assert p.is_anonymous is False


def test_resolve_with_unknown_key_raises_401(temp_security, monkeypatch):
    monkeypatch.setattr(settings.security, "api_keys", {"sk-alice": "alice"})
    with pytest.raises(HTTPException) as exc:
        _resolve("sk-bob")
    assert exc.value.status_code == 401
    assert "Invalid" in exc.value.detail


def test_resolve_anonymous_when_allowed(temp_security, monkeypatch):
    monkeypatch.setattr(settings.security, "api_keys", {})
    monkeypatch.setattr(settings.security, "allow_anonymous", True)
    p = _resolve(None)
    assert p.principal_id == "anonymous"
    assert p.is_anonymous is True


def test_resolve_without_key_raises_401_when_anonymous_forbidden(
    temp_security, monkeypatch
):
    monkeypatch.setattr(settings.security, "api_keys", {})
    monkeypatch.setattr(settings.security, "allow_anonymous", False)
    with pytest.raises(HTTPException) as exc:
        _resolve(None)
    assert exc.value.status_code == 401
    assert "required" in exc.value.detail.lower()


# ---------- SecuritySettings.api_keys 解析 ----------


def test_security_settings_api_keys_accepts_json_string():
    from config.settings import SecuritySettings

    cfg = SecuritySettings(API_KEYS='{"sk-a":"alice","sk-b":"bob"}')
    assert cfg.api_keys == {"sk-a": "alice", "sk-b": "bob"}


def test_security_settings_api_keys_empty_defaults_to_dict():
    from config.settings import SecuritySettings

    cfg = SecuritySettings(API_KEYS="")
    assert cfg.api_keys == {}


def test_security_settings_api_keys_rejects_invalid_json():
    from config.settings import SecuritySettings

    with pytest.raises(Exception):
        SecuritySettings(API_KEYS="{not json")


def test_security_settings_api_keys_rejects_non_object_json():
    from config.settings import SecuritySettings

    with pytest.raises(Exception):
        SecuritySettings(API_KEYS='["not","an","object"]')
