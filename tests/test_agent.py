from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from src.agent.agent import LegalContractAgent
from src.api.main import create_app
from src.api.routes._pipeline_qa import _agent_routing_enabled
from src.evaluation.metrics_store import MetricsStore
from src.pipeline.chat_scope_registry import ChatScopeRegistry


class _LLMMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, route_tool: str) -> None:
        self.route_tool = route_tool

    def invoke(self, messages):
        system_prompt = str(messages[0].content).lower()
        if "routing agent" in system_prompt:
            return _LLMMessage(
                json.dumps({"tool": self.route_tool, "reason": f"Route to {self.route_tool} for this test."})
            )
        return _LLMMessage("Grounded answer based on provided context. [chunk_id=c1]")


class _FakeContractRetriever:
    def __init__(self, return_empty: bool = False) -> None:
        self.return_empty = return_empty

    def get_top_k(
        self,
        query: str,
        contract_id: str | None = None,
        k: int = 5,
        clause_hints: list[str] | None = None,
    ):
        if self.return_empty:
            return []
        return [
            {
                "chunk_id": "c1",
                "text": "Indemnification is capped at direct damages.",
                "metadata": {
                    "contract_name": "sample_contract",
                    "clause_type": "indemnification",
                    "page_number": 4,
                },
                "fused_score": 0.12,
            }
        ]


def test_agent_routes_to_contract_search() -> None:
    agent = LegalContractAgent(clause_retriever=_FakeContractRetriever(), llm=_FakeLLM(route_tool="contract_search"))
    result = agent.run("What is the indemnification cap?")

    assert result["tool_used"] == "contract_search"
    assert result["source_chunks"]
    assert result["used_web_fallback"] is False


def test_agent_routes_to_web_search() -> None:
    agent = LegalContractAgent(clause_retriever=_FakeContractRetriever(), llm=_FakeLLM(route_tool="web_search"))
    result = agent.run("What is the market indemnity cap in SaaS deals?")

    assert result["tool_used"] == "web_search"


def test_agent_falls_back_to_web_when_contract_has_no_results() -> None:
    agent = LegalContractAgent(
        clause_retriever=_FakeContractRetriever(return_empty=True),
        llm=_FakeLLM(route_tool="contract_search"),
    )
    result = agent.run("Find the termination clause")

    assert result["used_web_fallback"] is True
    assert result["tool_used"] == "web_search"


# ---------------------------------------------------------------------------
# API integration tests for agent routing
# ---------------------------------------------------------------------------


class _FakeAPIEvaluator:
    def evaluate_single(self, question: str, answer: str, contexts: list[str], ground_truth: str = ""):
        return {
            "faithfulness": 0.9,
            "answer_relevance": 0.9,
            "context_precision": 0.9,
            "context_recall": 0.9,
        }


class _FakeAPIPipeline:
    def __init__(self) -> None:
        self._uploaded_contracts: list[dict[str, str | int]] = []

    def ask(
        self,
        question: str,
        contract_id: str | None = None,
        ground_truth: str = "",
        allowed_contract_ids: list[str] | None = None,
    ):
        return {
            "answer": "Pipeline answer.",
            "tool_used": "pipeline_contract_search",
            "route_reason": "Direct pipeline path.",
            "used_web_fallback": False,
            "matched_clause_hints": [],
            "sources": [],
            "evaluation": {
                "faithfulness": 0.9,
                "answer_relevance": 0.9,
                "context_precision": 0.9,
                "context_recall": 0.9,
            },
            "citations": [],
            "source_chunks": [],
        }

    def ingest_upload(self, filename: str, file_bytes: bytes, contract_id: str | None = None):
        return {"contract_id": contract_id or "fake", "chunks_ingested": 1, "message": "ok"}

    def list_contracts(self):
        return list(self._uploaded_contracts)

    def delete_contract(self, contract_id: str) -> None:
        pass


def _override_app_state(app, tmp_path, pipeline) -> None:
    db_path = tmp_path / "test_metrics.db"
    store = MetricsStore(database_url=f"sqlite:///{db_path}")
    store.init_db()
    app.state.metrics_store = store
    app.state.evaluator = _FakeAPIEvaluator()
    app.state.pipeline = pipeline
    app.state.chat_scope_registry = ChatScopeRegistry(registry_path=tmp_path / "chat_scope_registry.json")


class TestAgentRoutingIntegration:
    """Test that USE_AGENT_ROUTING env var controls the routing path."""

    def test_agent_routing_disabled_by_default(self, monkeypatch, tmp_path) -> None:
        """When USE_AGENT_ROUTING is not set, pipeline path is used."""
        monkeypatch.delenv("USE_AGENT_ROUTING", raising=False)
        assert _agent_routing_enabled() is False

        app = create_app()
        with TestClient(app) as client:
            _override_app_state(app, tmp_path, _FakeAPIPipeline())
            response = client.post("/query", json={"query": "What is termination?"})
            assert response.status_code == 200
            assert response.json()["tool_used"] == "pipeline_contract_search"

    def test_agent_routing_enabled_via_env(self, monkeypatch, tmp_path) -> None:
        """When USE_AGENT_ROUTING=1, the agent path is attempted."""
        monkeypatch.setenv("USE_AGENT_ROUTING", "1")
        assert _agent_routing_enabled() is True

    def test_agent_fallback_to_pipeline_on_error(self, monkeypatch, tmp_path) -> None:
        """When agent routing is enabled but agent creation fails, it falls
        back to the pipeline path."""
        monkeypatch.setenv("USE_AGENT_ROUTING", "1")
        app = create_app()
        with TestClient(app) as client:
            _override_app_state(app, tmp_path, _FakeAPIPipeline())
            # Pipeline attribute 'retriever' won't exist on _FakeAPIPipeline,
            # so agent creation will fail — it should fall back to pipeline.ask()
            response = client.post("/query", json={"query": "What is termination?"})
            assert response.status_code == 200
            assert response.json()["tool_used"] == "pipeline_contract_search"
