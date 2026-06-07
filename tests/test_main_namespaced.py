"""端到端：HTTP 路由层的命名空间隔离测试。

不调真实 LLM，把 service.chat / clear_chat_memory 用 monkeypatch 替换为捕获桩。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import services as services_module
from app.main import app
from config import settings


@pytest.fixture
def fresh_security(monkeypatch):
    """统一的 security 配置 fixture：alice 与 bob 各一个 key，允许匿名。"""
    monkeypatch.setattr(
        settings.security,
        "api_keys",
        {"sk-alice": "alice", "sk-bob": "bob"},
    )
    monkeypatch.setattr(settings.security, "allow_anonymous", True)


@pytest.fixture
def captured_chats(monkeypatch):
    """替换 chat() 与 clear_chat_memory()：捕获 effective chat_id，避免真调 LLM。"""
    captured: dict[str, list] = {"chats": [], "clears": []}

    def fake_chat(self, message, chat_id, use_memory=True):
        captured["chats"].append(
            {"message": message, "chat_id": chat_id, "use_memory": use_memory}
        )
        return f"echo:{chat_id}:{message}"

    def fake_clear(self, chat_id):
        captured["clears"].append(chat_id)

    monkeypatch.setattr(
        services_module.FinancialAdvisorService, "chat", fake_chat
    )
    monkeypatch.setattr(
        services_module.FinancialAdvisorService, "clear_chat_memory", fake_clear
    )
    return captured


@pytest.fixture
def client():
    return TestClient(app)


# ---------- 命名空间隔离 ----------


def test_chat_with_different_keys_yields_different_namespaces(
    fresh_security, captured_chats, client
):
    client.post(
        "/api/chat",
        json={"message": "hi", "chat_id": "default", "use_memory": True},
        headers={"X-API-Key": "sk-alice"},
    )
    client.post(
        "/api/chat",
        json={"message": "hi", "chat_id": "default", "use_memory": True},
        headers={"X-API-Key": "sk-bob"},
    )

    chat_ids = [c["chat_id"] for c in captured_chats["chats"]]
    assert chat_ids == ["alice:default", "bob:default"]


def test_chat_anonymous_falls_back_to_anonymous_namespace(
    fresh_security, captured_chats, client
):
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "chat_id": "default", "use_memory": True},
    )
    assert resp.status_code == 200
    assert captured_chats["chats"][0]["chat_id"] == "anonymous:default"


def test_chat_with_invalid_key_returns_401(fresh_security, captured_chats, client):
    resp = client.post(
        "/api/chat",
        json={"message": "hi"},
        headers={"X-API-Key": "sk-unknown"},
    )
    assert resp.status_code == 401
    assert captured_chats["chats"] == []  # 未进 service


def test_anonymous_forbidden_returns_401(monkeypatch, captured_chats, client):
    monkeypatch.setattr(settings.security, "api_keys", {})
    monkeypatch.setattr(settings.security, "allow_anonymous", False)

    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 401
    assert captured_chats["chats"] == []


# ---------- DELETE 隔离 ----------


def test_delete_memory_is_namespaced_per_principal(
    fresh_security, captured_chats, client
):
    """principal A 调 DELETE /api/memory/alice 实际操作 alice:alice，
    永远删不到 bob:alice。"""
    client.delete("/api/memory/alice", headers={"X-API-Key": "sk-alice"})
    client.delete("/api/memory/alice", headers={"X-API-Key": "sk-bob"})

    assert captured_chats["clears"] == ["alice:alice", "bob:alice"]


# ---------- /api/health 不受身份保护 ----------


def test_health_check_does_not_require_api_key(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------- chat_id 命名空间化保留响应字段 ----------


def test_chat_response_returns_raw_chat_id_not_namespaced(
    fresh_security, captured_chats, client
):
    """响应里返回客户端原 chat_id，不暴露 principal 前缀。"""
    resp = client.post(
        "/api/chat",
        json={"message": "hi", "chat_id": "foo"},
        headers={"X-API-Key": "sk-alice"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chat_id"] == "foo"  # 不带 "alice:" 前缀
