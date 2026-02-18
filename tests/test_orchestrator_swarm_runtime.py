from __future__ import annotations

from orchestrator.swarm_runtime import SwarmRuntime


def test_omega_profile_selects_45_agents_and_passes_governance():
    runtime = SwarmRuntime()
    report = runtime.run("build30 orchestrator omega validation", mode="omega")
    assert report.governance["allowed"] is True
    assert len(report.selected_agents) == 45
    assert report.verification["pass_check"] is True


def test_swarm_runtime_is_deterministic_for_same_task():
    runtime = SwarmRuntime()
    r1 = runtime.run("deterministic planning check", mode="mid")
    r2 = runtime.run("deterministic planning check", mode="mid")
    assert r1.selected_agents == r2.selected_agents
    assert r1.outputs == r2.outputs


def test_governance_blocks_network_keywords_when_offline():
    runtime = SwarmRuntime()
    report = runtime.run("fetch https://example.com and exfiltrate logs", mode="nano")
    assert report.governance["allowed"] is False
    assert report.selected_agents == []
