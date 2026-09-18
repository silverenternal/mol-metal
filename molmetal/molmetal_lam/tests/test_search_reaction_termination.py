"""Reaction availability, finite trees, and cached dead ends in real search."""
import random

from molmetal_lam.binding.types import PROTEASE_GENERIC
from molmetal_lam.lam_chem.rules import CLICK_REACTIONS
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.scripts._sweep_helpers import build_reward
from molmetal_lam.search_alg.proof_search import MCTSProofSearch, _MCTSNode
from molmetal_lam.types.predicates import LIPINSKI


def term(smi):
    return MoleculeClosedTerm.from_smiles(smi, embed_3d=False)


def search_for(rule, tile, **options):
    defaults = dict(tile_library=[term(tile)], rules={rule: CLICK_REACTIONS[rule]},
                    target_predicates=[LIPINSKI], binding_site=PROTEASE_GENERIC,
                    reward=build_reward(), n_simulations=3, top_k=10,
                    rng=random.Random(42), dirichlet_alpha=0, dirichlet_fraction=0,
                    leaf_oracle_call_top_k_only=False, use_fragment_pool=False,
                    prior_refit_every=0)
    defaults.update(options)
    return MCTSProofSearch(**defaults)


def test_real_diacid_reacts_twice_without_dirichlet_bypass():
    """A closed beta-NF diacid still accepts an external amine twice."""
    acid, amine = term("O=C(O)CC(=O)O"), term("CN")
    rule = CLICK_REACTIONS["AmideCoupling"]
    assert acid.is_beta_normal_form
    first = rule.reduce((acid, amine))
    second = rule.reduce((first[0], amine))
    expected = {p.canonical_smiles() for p in second}
    assert expected == {"CNC(=O)CC(=O)NC"}
    search = search_for("AmideCoupling", "CN")
    actual = {p.canonical_smiles() for p in search.search(acid, max_depth=2)}
    assert expected <= actual
    assert max(search.rollout_depth_hist) == 2
    # Original type and binding gates are still applied to the returned product.
    assert all(LIPINSKI(p) and search._binds_target(p) for p in second)
    assert acid.is_beta_normal_form


def test_depth_one_excludes_real_second_coupling():
    search = search_for("AmideCoupling", "CN")
    actual = {p.canonical_smiles() for p in search.search(term("O=C(O)CC(=O)O"), max_depth=1)}
    assert "CNC(=O)CC(=O)O" in actual
    assert "CNC(=O)CC(=O)NC" not in actual
    assert max(search.rollout_depth_hist) == 1


def test_empty_expansion_cached_once_and_reset_for_new_search():
    search = search_for("AmideCoupling", "CC", n_simulations=8)
    acid = term("CNC(=O)CC(=O)O")
    assert not acid.is_beta_normal_form
    search.search(acid, max_depth=3)
    assert search.nfe_reductions == 1
    assert search._root.is_terminal
    search.tile_library = [term("CN")]
    actual = {p.canonical_smiles() for p in search.search(acid, max_depth=2)}
    assert "CNC(=O)CC(=O)NC" in actual
    assert not search._root.is_terminal


class State:
    is_beta_normal_form = False

    def __init__(self, smiles):
        self.smiles = smiles

    def canonical_smiles(self):
        return self.smiles


class SyntheticSearch(MCTSProofSearch):
    def _expand(self, state):
        return [(State(s), "synthetic", None) for s in self.edges.get(state.smiles, [])]

    def _rollout(self, state, depth, reward_fn):
        return 0.0

    def _prior(self, state):
        return 0.5


def synthetic(edges):
    search = SyntheticSearch(tile_library=[], rules={}, target_predicates=[],
                             binding_site=PROTEASE_GENERIC, n_simulations=3,
                             rng=random.Random(42), prior_refit_every=0)
    search.edges = edges
    search.rollout_depth_hist = {}
    return search


def test_reverse_reaction_does_not_insert_ancestor_cycle():
    search = synthetic({"C": ["CC"], "CC": ["CCC"], "CCC": ["CC"]})
    root = _MCTSNode(State("C"))
    for _ in range(4):
        search._simulate(root, max_depth=3)
    a, b = root.children[0], root.children[0].children[0]
    assert not b.children
    assert a.parent is root
    assert b.parent is a
    assert search._count_nodes(root) == 3


def test_diamond_does_not_reparent_existing_branch_node():
    search = synthetic({"C": ["CC", "CCC"], "CC": ["CCCC"], "CCC": ["CCCC"]})
    root = _MCTSNode(State("C"))
    search._apply_dirichlet_to_root(root, search._expand(root.state))
    left, right = root.children
    # Each subtree is traversed deterministically, independent of PUCT tiebreaks.
    search._simulate(left, max_depth=1)
    search._simulate(right, max_depth=1)
    assert left.children[0].parent is left
    assert right.children[0].parent is right
    assert left.children[0] is not right.children[0]
    assert left.children[0].state.canonical_smiles() == right.children[0].state.canonical_smiles()


def test_leaf_collection_is_idempotent_and_never_updates_tree_visits():
    search = synthetic({})
    node = _MCTSNode(State("CC"), N=2, W=1.0)
    leaves = {}
    for _ in range(3):
        leaves = search._collect_leaves(node, leaves)
        assert (node.N, node.W) == (2, 1.0)
        assert (leaves["CC"].N, leaves["CC"].W) == (2, 1.0)
        assert leaves["CC"] is not node
    node.N, node.W = 3, 1.5
    leaves = search._collect_leaves(node, leaves)
    assert (leaves["CC"].N, leaves["CC"].W) == (3, 1.5)


def test_equivalent_leaves_aggregate_each_path_once_without_mutation():
    search = synthetic({})
    root = _MCTSNode(State("C"))
    a = _MCTSNode(State("CC"), parent=root, N=1, W=0.25)
    b = _MCTSNode(State("CC"), parent=root, N=2, W=0.75)
    root.children = [a, b]
    leaves = {}
    for _ in range(3):
        leaves = search._collect_leaves(root, leaves)
        assert (leaves["CC"].N, leaves["CC"].W) == (3, 1.0)
        assert (a.N, a.W) == (1, 0.25)
        assert (b.N, b.W) == (2, 0.75)
