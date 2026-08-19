"""Continuity-first wake policy regression coverage."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import frontmatter
import pytest

import tools._runtime as rt
from tools.breath.search import surface_search
from tools.breath.surface import (
    _select_wake_candidates,
    _stable_wake_partition,
    _wake_bucket_class,
    _wake_policy_config,
    surface_default,
)


class DisabledEmbedding:
    enabled = False

    async def search_similar(self, query, top_k=20):
        return []


def _install_runtime(bucket_mgr, decay_eng, wake=None):
    surfacing = {} if wake is None else {"wake": wake}
    rt.config = {"surfacing": surfacing}
    rt.bucket_mgr = bucket_mgr
    rt.decay_engine = decay_eng
    rt.embedding_engine = DisabledEmbedding()
    rt.logger = MagicMock()
    rt.fire_webhook = None
    rt.mark_op = None


def _bucket(bucket_id, domain=None, tags=None):
    return {
        "id": bucket_id,
        "content": bucket_id,
        "metadata": {
            "type": "dynamic",
            "domain": domain or [],
            "tags": tags or [],
        },
    }


def _patch_meta(bucket_mgr, bucket_id, **updates):
    path = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(path)
    for key, value in updates.items():
        post[key] = value
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(frontmatter.dumps(post))


def test_wake_defaults_enabled_and_normalizes_domain_plus_tags():
    cfg = _wake_policy_config({})

    assert cfg["enabled"] is True
    assert cfg["pure_work_max"] == 1
    assert _wake_bucket_class(_bucket("work", domain=["WORK"]), cfg) == "pure_work"
    assert (
        _wake_bucket_class(
            _bucket("work-voice", domain=["work"], tags=["Voice"]),
            cfg,
        )
        == "continuity"
    )
    assert (
        _wake_bucket_class(
            _bucket(
                "work-relationship",
                domain="work",
                tags="relationship",
            ),
            cfg,
        )
        == "continuity"
    )
    assert (
        _wake_bucket_class(_bucket("ordinary", domain=["milestone"]), cfg)
        == "ordinary"
    )


def test_wake_partition_is_stable_and_quota_runs_before_result_cap():
    cfg = _wake_policy_config({"wake": {"pure_work_max": 1}})
    candidates = [
        _bucket("w1", domain=["work"]),
        _bucket("o1", domain=["milestone"]),
        _bucket("c1", domain=["continuity"]),
        _bucket("w2", domain=["work"]),
        _bucket("c2", tags=["voice"]),
        _bucket("o2", domain=["memory"]),
    ]

    ordered = _stable_wake_partition(candidates, cfg)
    assert [row["id"] for row in ordered] == [
        "c1",
        "c2",
        "o1",
        "o2",
        "w1",
        "w2",
    ]

    selected = _select_wake_candidates(
        candidates,
        max_results=5,
        wake_cfg=cfg,
    )
    assert [row["id"] for row in selected] == [
        "c1",
        "c2",
        "o1",
        "o2",
        "w1",
    ]


def test_disabled_wake_policy_preserves_legacy_candidate_order():
    cfg = _wake_policy_config(
        {"wake": {"enabled": False, "pure_work_max": 0}}
    )
    candidates = [
        _bucket("w1", domain=["work"]),
        _bucket("c1", domain=["continuity"]),
        _bucket("o1", domain=["memory"]),
    ]

    assert _stable_wake_partition(candidates, cfg) == candidates
    selected = _select_wake_candidates(
        candidates,
        max_results=2,
        wake_cfg=cfg,
    )
    assert [row["id"] for row in selected] == ["w1", "c1"]


@pytest.mark.asyncio
async def test_default_breath_is_continuity_first_caps_pure_work_and_bypasses_pinned(
    bucket_mgr,
    decay_eng,
    monkeypatch,
):
    monkeypatch.setattr(
        decay_eng,
        "calculate_score",
        lambda meta: float(meta.get("importance") or 1),
    )
    monkeypatch.setattr(
        "tools.breath.surface.random.shuffle",
        lambda rows: None,
    )
    monkeypatch.setattr(
        "tools.breath.surface.random.random",
        lambda: 1.0,
    )
    _install_runtime(bucket_mgr, decay_eng)

    pinned_id = await bucket_mgr.create(
        content="PINNED-WORK-CONTROL",
        pinned=True,
        domain=["work"],
    )
    work_one = await bucket_mgr.create(
        content="PURE-WORK-ONE",
        importance=7,
        domain=["work"],
    )
    work_two = await bucket_mgr.create(
        content="PURE-WORK-TWO",
        importance=6,
        domain=["work"],
    )
    ordinary_id = await bucket_mgr.create(
        content="ORDINARY-MILESTONE",
        importance=5,
        domain=["milestone"],
    )
    continuity_id = await bucket_mgr.create(
        content="RELATIONSHIP-CONTINUITY",
        importance=4,
        domain=["relationship"],
    )

    result = await surface_default(
        max_results=3,
        max_tokens=20_000,
        tag_filter=[],
    )

    assert pinned_id in result
    assert continuity_id in result
    assert ordinary_id in result
    assert work_one in result
    assert work_two not in result
    assert (
        result.index("RELATIONSHIP-CONTINUITY")
        < result.index("ORDINARY-MILESTONE")
        < result.index("PURE-WORK-ONE")
    )


@pytest.mark.asyncio
async def test_passive_and_resolved_share_remaining_pure_work_quota(
    bucket_mgr,
    decay_eng,
    monkeypatch,
):
    monkeypatch.setattr(
        decay_eng,
        "calculate_score",
        lambda meta: float(meta.get("importance") or 1),
    )
    monkeypatch.setattr(
        "tools.breath.surface.random.shuffle",
        lambda rows: None,
    )
    monkeypatch.setattr(
        "tools.breath.surface.random.random",
        lambda: 0.0,
    )
    _install_runtime(
        bucket_mgr,
        decay_eng,
        wake={"pure_work_max": 1},
    )

    continuity_id = await bucket_mgr.create(
        content="CONTINUITY-MAIN",
        importance=9,
        domain=["continuity"],
    )
    main_work_id = await bucket_mgr.create(
        content="PURE-WORK-MAIN",
        importance=10,
        domain=["work"],
    )
    passive_work_id = await bucket_mgr.create(
        content="PURE-WORK-PASSIVE",
        importance=9,
        domain=["work"],
    )
    resolved_work_id = await bucket_mgr.create(
        content="PURE-WORK-RESOLVED",
        importance=10,
        domain=["work"],
    )

    active_ts = datetime.now().isoformat()
    old_ts = (datetime.now() - timedelta(days=8)).isoformat()
    _patch_meta(
        bucket_mgr,
        continuity_id,
        activation_count=1,
        last_active=active_ts,
    )
    _patch_meta(
        bucket_mgr,
        main_work_id,
        activation_count=1,
        last_active=active_ts,
    )
    _patch_meta(
        bucket_mgr,
        passive_work_id,
        activation_count=1,
        last_active=old_ts,
    )
    await bucket_mgr.update(resolved_work_id, resolved=True)

    result = await surface_default(
        max_results=2,
        max_tokens=20_000,
        tag_filter=[],
    )

    assert continuity_id in result
    assert main_work_id in result
    assert passive_work_id not in result
    assert resolved_work_id not in result
    assert "PURE-WORK-PASSIVE" not in result
    assert "PURE-WORK-RESOLVED" not in result


@pytest.mark.asyncio
async def test_explicit_search_ignores_wake_pure_work_quota(
    bucket_mgr,
    decay_eng,
):
    marker = "WAKE-SEARCH-PURE-WORK-42B7"
    first = await bucket_mgr.create(
        content=f"{marker} first explicit work memory.",
        importance=8,
        domain=["work"],
    )
    second = await bucket_mgr.create(
        content=f"{marker} second explicit work memory.",
        importance=7,
        domain=["work"],
    )
    _install_runtime(
        bucket_mgr,
        decay_eng,
        wake={"pure_work_max": 0},
    )

    result = await surface_search(
        query=marker,
        max_results=5,
        max_tokens=20_000,
        domain="",
        valence=-1,
        arousal=-1,
        tag_filter=[],
    )

    assert first in result
    assert second in result
