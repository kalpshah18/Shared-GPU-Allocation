"""Verify observers against actual runs and intervention/accounting identities."""
import json
from pathlib import Path
import numpy as np
import pytest

from analysis.failure_diagnostics import karma_statistics, karma_trace, rank_trace, service_totals
from sim.adaptation import ProfileEvaluator, best_response_dynamics
from sim.config import Config
from sim.environment import SeedPackage
from sim.mechanisms import KarmaCapMechanism, RankCapMechanism
from sim.policies import get_policy
from sim.runner import run_mixed, run_rollout

ROOT = Path(__file__).resolve().parent.parent


def test_followup_evaluation_seeds_are_independent():
    fresh = json.loads((ROOT/'seeds/followup_seeds.json').read_text())['seeds']
    excluded = list(range(101,106))+list(range(201,206))
    for name in ('master_seeds.json','extension_seeds.json'):
        excluded += json.loads((ROOT/'seeds'/name).read_text())['seeds']
    assert len(set(fresh)) == 30 and not set(fresh) & set(excluded)


@pytest.mark.parametrize('updating', [True, False])
def test_rank_observer_preserves_rollout_actions_and_payoffs(updating):
    cfg = Config(n=4,k=1,T=10,wait_cap=3,rank_rollout_update=updating)
    pkg = SeedPackage.generate(757622112,cfg)
    actual, reports = run_rollout(RankCapMechanism(cfg),pkg,cfg,0,None,
                                 np.random.default_rng([pkg.master_seed,159,4,6]),H=4,n_rollouts=6)
    observed, trace = rank_trace(pkg,cfg,H=4,R=6)
    np.testing.assert_array_equal(actual.allocations,observed.allocations)
    np.testing.assert_array_equal(reports,[r['report'] for r in trace])
    assert service_totals(trace)['utility'] == pytest.approx(np.sum(pkg.valuations[0]*np.array(actual.allocations)[:,0]))


def test_truth_reference_replay_removes_rank_shift_and_preserves_truthful_baseline():
    cfg = Config(n=3,k=1,T=20,wait_cap=3)
    pkg = SeedPackage.generate(21,cfg)
    a,_ = rank_trace(pkg,cfg,reports=pkg.valuations[0])
    b,baseline = rank_trace(pkg,cfg,reports=pkg.valuations[0],truth_reference=True)
    np.testing.assert_array_equal(a.allocations,b.allocations)
    _,contaminated = rank_trace(pkg,cfg,reports=np.zeros(cfg.T))
    _,clean = rank_trace(pkg,cfg,reports=np.zeros(cfg.T),truth_reference=True)
    assert any(r['history_rank_shift'] > 0 for r in contaminated[1:])
    assert all(r['history_rank_shift'] == 0 for r in clean+baseline)


@pytest.mark.parametrize('cap',[False,True])
def test_karma_observer_preserves_outcomes_and_exact_credit_service_accounting(cap):
    cfg = Config(n=5,k=2,T=20,wait_cap=2,enable_wait_cap=cap,karma_init=.3)
    pkg = SeedPackage.generate(22,cfg)
    observed,traces = karma_trace(pkg,cfg,[0,1,3])
    real = run_mixed(KarmaCapMechanism(cfg),pkg,cfg,get_policy('cap_2'),np.array([0,1,3]))
    np.testing.assert_array_equal(real.allocations,observed.allocations)
    for trace in traces.values():
        stats = karma_statistics(trace)
        assert stats['forced_utility']+stats['auction_utility'] == pytest.approx(stats['utility'])
        assert stats['forced_credit_spent']+stats['auction_credit_spent'] == pytest.approx(stats['credit_spent'])
        for row in trace:
            assert row['next_balance'] == pytest.approx(row['balance']-row['charge']+row['dividend']-row['ceiling_loss'])
    if cap:
        assert any(r['forced'] and r['charge']>0 for trace in traces.values() for r in trace)


def test_recurrence_continuation_keeps_searching_a_game_without_a_pure_fixed_point():
    # Matching pennies: each sweep has a profitable response, and a repeated
    # boundary profile should be recorded rather than mistaken for convergence.
    class ToyGame:
        cfg = Config(n=2,k=1,T=1)
        names = ('a','b')
        table = {(0,0):(1,0), (0,1):(0,1), (1,0):(0,1), (1,1):(1,0)}
        cache = table
        def utility(self,profile):
            return np.array(self.table[tuple(profile)],dtype=float)
        def gains(self,profile):
            base=self.utility(profile)
            gains=np.zeros((2,2))
            for u in range(2):
                for p in range(2):
                    dev=list(profile);dev[u]=p
                    gains[u,p]=self.utility(dev)[u]-base[u]
            return gains
    short=best_response_dynamics(ToyGame(),[0,0],8)
    long=best_response_dynamics(ToyGame(),[0,0],8,stop_on_recurrence=False)
    assert short['status']=='cycle' and short['sweeps'] < 8
    assert long['status']=='sweep_limit' and long['sweeps']==8
    assert long['training_exploitability']==1 and long['recurrent_sweeps']


@pytest.mark.parametrize('exp',[16,17,18])
def test_followup_experiments_write_seed_roles_traces_and_strict_json(exp,tmp_path):
    from experiments.e16_rank_history import run_e16
    from experiments.e17_karma_accounting import run_e17
    from experiments.e18_extended_adaptation import run_e18
    options=dict(results_dir=str(tmp_path),n_bootstrap=30,verbose=False)
    if exp==16:
        run_e16([757622112,386682252],n=4,T=8,H=3,R=4,**options)
    elif exp==17:
        run_e17([757622112,386682252],n=4,T=10,**options)
    else:
        run_e18([30001,30002],n=3,T=8,budgets=(1,2),training_seeds=[201],
                specs=[('Vickrey','VickreyMechanism',{})],**options)
    folder=tmp_path/f'e{exp}'
    assert (folder/'summary.json').exists() and (folder/'paired.json').exists()
    cfg=json.loads((folder/'config.json').read_text())
    assert cfg['seed_role']==('heldout' if exp==18 else 'descriptive_reanalysis')
    for path in folder.rglob('*.json'):
        json.loads(path.read_text(),parse_constant=lambda value:pytest.fail(value))
    from analysis.failure_report import run_figures
    run_figures(str(tmp_path),str(tmp_path/'figures'))
    assert list((tmp_path/'figures').glob('*.pdf'))
    from analysis.report import build_report
    assert f'## E{exp}' in build_report(str(tmp_path))
