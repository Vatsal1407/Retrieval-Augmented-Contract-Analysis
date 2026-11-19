from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from src.api.schemas import DeleteResponse

router = APIRouter(tags=["delete"])


def _strict_scope_enabled() -> bool:
    return os.getenv("REQUIRE_CHAT_SCOPE", "1").strip().lower() in {"1", "true", "yes"}


@router.delete("/contracts/{contract_id}", response_model=DeleteResponse)
async def delete_contract(
    contract_id: str,
    request: Request,
    chat_id: str = Query(..., min_length=1),
) -> DeleteResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized.")

    chat_scope_registry = getattr(request.app.state, "chat_scope_registry", None)
    if chat_scope_registry is None:
        raise HTTPException(status_code=503, detail="Chat scope registry is not initialized.")

    allowed_contract_ids = set(chat_scope_registry.list_contract_ids(chat_id))
    if contract_id not in allowed_contract_ids:
        raise HTTPException(status_code=403, detail="Contract is not accessible from this chat.")

    try:
        await run_in_threadpool(pipeline.delete_contract, contract_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Contract deletion failed.")

    return DeleteResponse(
        contract_id=contract_id,
        message="Contract deleted successfully.",
    )
