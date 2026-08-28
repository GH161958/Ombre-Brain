from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from typing import Any

from tools.plan.core import is_letter_bucket, letter_lock_state  # type: ignore
from utils import parse_bool  # type: ignore


MEMORY_RESOURCE_SCHEMA = "ombre-memory-resource/v0"
MEMORY_RESOURCE_URI_TEMPLATE = "ombre://memory/{bucket_id}"
_LOCKED_LETTER_NAME = "一封上锁的信"
_LOCKED_LETTER_DOMAIN = ["letter"]
_LOCKED_LETTER_TAGS = ["__letter__"]
_MAX_BUCKET_ID_CHARS = 256


class InvalidMemoryBucketId(ValueError):
    """The resource request did not contain a safe exact bucket ID."""


class MemoryBucketNotFound(LookupError):
    """No active Memory bucket exists for the requested exact ID."""


def _normalize_bucket_id(bucket_id: object) -> str:
    if not isinstance(bucket_id, str):
        raise InvalidMemoryBucketId("bucket_id must be a non-empty string")

    normalized = unicodedata.normalize("NFC", bucket_id).strip()
    if not normalized:
        raise InvalidMemoryBucketId("bucket_id must be a non-empty string")
    if len(normalized) > _MAX_BUCKET_ID_CHARS:
        raise InvalidMemoryBucketId("bucket_id is too long")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise InvalidMemoryBucketId("bucket_id must be an exact bucket identifier")
    if any(unicodedata.category(char) == "Cc" for char in normalized):
        raise InvalidMemoryBucketId("bucket_id contains control characters")
    return normalized


def _metadata_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[object] = value.split(",")
    elif isinstance(value, (list, tuple)):
        values = value
    elif isinstance(value, (set, frozenset)):
        values = sorted(value, key=str)
    else:
        values = (value,)
    return [str(item).strip() for item in values if str(item).strip()]


def _metadata_bool(metadata: dict[str, Any], name: str) -> bool:
    return parse_bool(metadata.get(name), default=False)


def _ordinary_payload(bucket: dict[str, Any], bucket_id: str) -> dict[str, Any]:
    metadata = bucket.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    lock_state = letter_lock_state(bucket, None) if is_letter_bucket(bucket) else {
        "lock_type": "none",
        "unlock_date": None,
    }
    content = bucket.get("content", "")
    if not isinstance(content, str):
        content = str(content or "")
    return {
        "schema": MEMORY_RESOURCE_SCHEMA,
        "authority": "ombre-brain",
        "id": str(bucket.get("id") or bucket_id),
        "name": str(metadata.get("name") or ""),
        "type": str(metadata.get("type") or "dynamic"),
        "domain": _metadata_list(metadata.get("domain")),
        "tags": _metadata_list(metadata.get("tags")),
        "content": content,
        "content_available": True,
        "importance": metadata.get("importance"),
        "resolved": _metadata_bool(metadata, "resolved"),
        "pinned": _metadata_bool(metadata, "pinned"),
        "digested": _metadata_bool(metadata, "digested"),
        "dont_surface": _metadata_bool(metadata, "dont_surface"),
        "created": metadata.get("created") or None,
        "last_active": metadata.get("last_active") or None,
        "letter_locked": False,
        "lock_type": str(lock_state.get("lock_type") or "none"),
        "unlock_date": lock_state.get("unlock_date") or None,
    }


def _locked_letter_payload(
    bucket: dict[str, Any], bucket_id: str, lock_state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": MEMORY_RESOURCE_SCHEMA,
        "authority": "ombre-brain",
        "id": str(bucket.get("id") or bucket_id),
        "name": _LOCKED_LETTER_NAME,
        "type": "letter",
        "domain": list(_LOCKED_LETTER_DOMAIN),
        "tags": list(_LOCKED_LETTER_TAGS),
        "content": "",
        "content_available": False,
        "importance": None,
        "resolved": False,
        "pinned": False,
        "digested": False,
        "dont_surface": True,
        "created": None,
        "last_active": None,
        "letter_locked": True,
        "lock_type": str(lock_state.get("lock_type") or "none"),
        "unlock_date": lock_state.get("unlock_date") or None,
    }


async def read_memory_resource(bucket_manager: Any, bucket_id: object) -> str:
    """Return one active Memory bucket as deterministic, read-only JSON text."""

    normalized_id = _normalize_bucket_id(bucket_id)
    bucket = await bucket_manager.get(normalized_id)
    if bucket is None:
        raise MemoryBucketNotFound(f"Memory bucket not found: {normalized_id}")
    if not isinstance(bucket, dict):
        raise MemoryBucketNotFound(f"Memory bucket not found: {normalized_id}")

    lock_state = letter_lock_state(bucket, None) if is_letter_bucket(bucket) else None
    if lock_state and lock_state.get("locked"):
        payload = _locked_letter_payload(bucket, normalized_id, lock_state)
    else:
        payload = _ordinary_payload(bucket, normalized_id)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
