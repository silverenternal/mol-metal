"""Contract tests for AiZynth 4.x collection and reaction-tree semantics."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from molmetal.molmetal_lam.sbdd_env.aizynth_adapter import AiZynthAdapter


def tree(solved, depths, stocked):
    leaves = [SimpleNamespace(smiles=f"leaf{i}") for i in range(len(depths))]
    return SimpleNamespace(
        is_solved=solved, leafs=lambda: iter(leaves),
        depth=lambda leaf: depths[leaves.index(leaf)],
        in_stock=lambda leaf: stocked[leaves.index(leaf)],
    )


def adapter_with_routes(trees):
    adapter = AiZynthAdapter()
    adapter._finder = SimpleNamespace(
        tree_search=Mock(), build_routes=Mock(),
        routes=SimpleNamespace(reaction_trees=trees),
    )
    return adapter


def test_partially_expanded_route_is_not_synthesis_success():
    report = adapter_with_routes([tree(False, [2, 4], [True, False])]).check("CCO")
    assert report.engine == "aizynthfinder"
    assert not report.synthesizable
    assert report.depth == 2
    assert report.route_smiles == ("leaf0",)


def test_select_solved_route_and_measure_longest_path_not_reaction_count():
    report = adapter_with_routes([
        tree(False, [2], [False]), tree(True, [2, 4, 4], [True, True, True]),
    ]).check("CCO")
    assert report.engine == "aizynthfinder"
    assert report.synthesizable and report.depth == 2
    assert len(report.route_smiles) == 3


def test_stock_target_zero_step_route_can_be_solved():
    report = adapter_with_routes([tree(True, [0], [True])]).check("CCO")
    assert report.synthesizable and report.depth == 0


def test_empty_route_collection():
    adapter = AiZynthAdapter()
    adapter._finder = SimpleNamespace(tree_search=Mock(), build_routes=Mock(), routes=[])
    report = adapter.check("CCO")
    assert report.engine == "aizynthfinder" and not report.synthesizable


@pytest.mark.parametrize("limits", [{"max_iterations": 0}, {"time_limit_s": 0}])
def test_reject_nonpositive_limits(limits):
    with pytest.raises(ValueError):
        AiZynthAdapter(**limits)


def test_load_selects_models_stock_and_propagates_budget(monkeypatch, tmp_path):
    from molmetal.molmetal_lam.sbdd_env import aizynth_adapter as module

    def collection(items):
        class Collection(SimpleNamespace):
            def __getitem__(self, key):
                return SimpleNamespace(model=SimpleNamespace())
        result = Collection(items=items, selection=[])
        result.select_first = lambda: setattr(result, "selection", items[:1])
        result.select_all = lambda: setattr(result, "selection", list(items))
        return result

    finder = SimpleNamespace(
        expansion_policy=collection(["uspto", "ringbreaker"]),
        stock=collection(["zinc"]), filter_policy=collection([]),
        config=SimpleNamespace(search=SimpleNamespace(iteration_limit=100, time_limit=120)),
    )
    monkeypatch.setattr(module, "_have_aizynthfinder", lambda: True)
    monkeypatch.setitem(sys.modules, "aizynthfinder.aizynthfinder", SimpleNamespace(AiZynthFinder=lambda **_: finder))
    config = tmp_path / "config.yml"
    config.write_text("{}")
    adapter = AiZynthAdapter(str(config), max_iterations=7, time_limit_s=3)
    assert adapter._finder is finder
    assert finder.config.search.iteration_limit == 7
    assert finder.config.search.time_limit == 3
    assert adapter.get_metadata()["expansion_policy"] == ["uspto"]
    assert adapter.get_metadata()["stock"] == ["zinc"]
    finder.expansion_policy.items = []
    unavailable = AiZynthAdapter(str(config))
    assert unavailable._finder is None
    assert "trained expansion policy" in unavailable._load_error
