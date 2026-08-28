import copy
import json

import pytest

from ombrebrain.protocol.memory_resource import (
    InvalidMemoryBucketId,
    MemoryBucketNotFound,
    read_memory_resource,
)


class ExactBucketManager:
    def __init__(self, bucket):
        self.bucket = bucket
        self.lookups = []

    async def get(self, bucket_id):
        self.lookups.append(bucket_id)
        if self.bucket is None or self.bucket.get("id") != bucket_id:
            return None
        return self.bucket


def _ordinary_bucket():
    return {
        "id": "abc123def456",
        "metadata": {
            "name": "An exact memory",
            "type": "dynamic",
            "domain": ["work", "design"],
            "tags": ["exact", "raw"],
            "importance": 8,
            "resolved": False,
            "pinned": True,
            "digested": False,
            "dont_surface": False,
            "created": "2026-08-28T09:00:00+00:00",
            "last_active": "2026-08-28T10:00:00+00:00",
        },
        "content": "# Exact Markdown\n\nKeep [[wikilinks]], *formatting*, and trailing space. \n",
        "path": "/not/read/by/resource.md",
    }


@pytest.mark.asyncio
async def test_exact_bucket_returns_exact_stored_markdown_without_mutation():
    bucket = _ordinary_bucket()
    before = copy.deepcopy(bucket)
    manager = ExactBucketManager(bucket)

    first = await read_memory_resource(manager, "  abc123def456  ")
    second = await read_memory_resource(manager, "abc123def456")
    payload = json.loads(first)

    assert first == second
    assert payload["schema"] == "ombre-memory-resource/v0"
    assert payload["authority"] == "ombre-brain"
    assert payload["id"] == "abc123def456"
    assert payload["content"] == before["content"]
    assert payload["content_available"] is True
    assert payload["letter_locked"] is False
    assert bucket == before


@pytest.mark.asyncio
async def test_locked_letter_redacts_body_and_all_private_metadata():
    secrets = {
        "title": "private-title-marker",
        "body": "private-body-marker",
        "tag": "private-tag-marker",
        "domain": "private-domain-marker",
        "author": "private-author-marker",
        "name": "private-name-marker",
        "created": "1843-private-date-marker",
    }
    bucket = {
        "id": "locked-letter-1",
        "metadata": {
            "name": secrets["name"],
            "title": secrets["title"],
            "type": "letter",
            "domain": [secrets["domain"]],
            "tags": ["__letter__", secrets["tag"]],
            "author": secrets["author"],
            "writer_name": "private-writer-marker",
            "importance": 10,
            "created": secrets["created"],
            "last_active": "1844-private-active-marker",
            "lock_type": "timed",
            "unlock_date": "2999-01-02T03:04:05+00:00",
            "locked_by": "human",
        },
        "content": secrets["body"],
    }
    before = copy.deepcopy(bucket)

    encoded = await read_memory_resource(ExactBucketManager(bucket), bucket["id"])
    payload = json.loads(encoded)

    assert payload["name"] == "一封上锁的信"
    assert payload["domain"] == ["letter"]
    assert payload["tags"] == ["__letter__"]
    assert payload["content"] == ""
    assert payload["content_available"] is False
    assert payload["letter_locked"] is True
    assert payload["lock_type"] == "timed"
    assert payload["unlock_date"] == "2999-01-02T03:04:05+00:00"
    assert payload["created"] is None
    assert payload["last_active"] is None
    assert all(secret not in encoded for secret in secrets.values())
    assert "private-writer-marker" not in encoded
    assert bucket == before


@pytest.mark.asyncio
async def test_missing_bucket_fails_explicitly():
    manager = ExactBucketManager(None)

    with pytest.raises(MemoryBucketNotFound, match="missing-id"):
        await read_memory_resource(manager, "missing-id")

    assert manager.lookups == ["missing-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bucket_id",
    [None, "", "   ", ".", "..", "../memory", "folder/memory", "folder\\memory", "bad\x00id"],
)
async def test_invalid_bucket_id_fails_before_storage_lookup(bucket_id):
    manager = ExactBucketManager(None)

    with pytest.raises(InvalidMemoryBucketId):
        await read_memory_resource(manager, bucket_id)

    assert manager.lookups == []


@pytest.mark.asyncio
async def test_server_registers_resource_template_without_adding_public_tool():
    import server

    templates = await server.mcp.list_resource_templates()
    template_uris = {
        str(
            getattr(
                template,
                "uri_template",
                getattr(template, "uriTemplate", ""),
            )
        )
        for template in templates
    }
    assert "ombre://memory/{bucket_id}" in template_uris

    tools = await server.mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    assert len(tool_names) == 24
    assert "memory_read" not in tool_names
    assert "memory_read_structured" not in tool_names
