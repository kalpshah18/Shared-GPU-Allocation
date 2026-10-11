"""Audit follow-up outputs, trajectory identities and original-run reproduction."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np

from analysis.failure_diagnostics import karma_statistics, service_totals
from experiments.extensions import ROOT
from sim.environment import save_json


def read(path):
    def reject(value):
        raise AssertionError(f'Non-finite JSON in {path}: {value}')
    return json.loads(Path(path).read_text(),parse_constant=reject)


def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-9), (a,b)


def audit(results_dir='results'):
    rd=Path(results_dir)
    config={exp:read(rd/exp/'config.json') for exp in ('e16','e17','e18')}
    all_old=[]
    for name in ('master_seeds.json','extension_seeds.json'):
        all_old+=read(ROOT/'seeds'/name)['seeds']
    assert not set(config['e18']['seeds']) & set(all_old+list(range(101,106))+list(range(201,206)))
    assert not set(config['e18']['seeds']) & set(config['e18']['training_seeds'])
    assert config['e16']['seeds']==config['e17']['seeds']
    assert config['e16']['seed_role']==config['e17']['seed_role']=='descriptive_reanalysis'
    assert config['e18']['seed_role']=='heldout'
    counts,lookup={},{}
    mean_checks=0
    for exp in config:
        summaries=read(rd/exp/'summary.json')
        for summary in summaries:
            if exp=='e16':
                cell=summary['mode'];key=(exp,cell)
            elif exp=='e17':
                cell=f"{summary['label']}_rho{summary['rho']:g}";key=(exp,summary['rho'],summary['label'])
            else:
                cell=f"{summary['label']}_{summary['initial']}_budget{summary['budget']}"
                key=(exp,summary['label'],summary['initial'],summary['budget'])
            source=read(rd/exp/'raw'/f'{cell}.json')
            rows=source['rows'];lookup[key]=rows
            assert [r['seed'] for r in rows]==config[exp]['seeds']
            assert len(rows)==summary['n_seeds']==config[exp]['n_seeds']
            assert source['config']['n_bootstrap']==config[exp]['n_bootstrap']
            for metric,stat in summary.items():
                if not isinstance(stat,dict) or 'mean' not in stat or metric=='restricted_exploitability':continue
                close(stat['mean'],np.mean([r[metric] for r in rows]));mean_checks+=1
            if exp=='e16':
                for row in rows:
                    trace=read(rd/exp/'traces'/f"{cell}_{row['seed']}.json")
                    base,attack,replay=[service_totals(trace[name]) for name in ('baseline','attack','clean_reference')]
                    close(row['M_rollout'],attack['utility']-base['utility'])
                    close(row['forced_gain']+row['auction_gain'],row['M_rollout'])
                    close(row['ledger_effect'],attack['utility']-replay['utility'])
                    assert all(r['history_rank_shift']==0 for r in trace['baseline']+trace['clean_reference'])
                    assert [r['report'] for r in trace['attack']]==[r['report'] for r in trace['clean_reference']]
            elif exp=='e17':
                for row in rows:
                    close(row['M_uni'],row['forced_gain']+row['auction_gain'])
                    for item in row['individual']:
                        close(item['M_uni'],item['forced_gain']+item['auction_gain'])
                        close(item['credit_spent_change'],item['forced_credit_spent_change']+item['auction_credit_spent_change'])
            else:
                means=np.mean([r['unilateral_gains'] for r in rows],axis=0)
                close(summary['restricted_exploitability']['mean'],max(0,means.max()))
                np.testing.assert_allclose(summary['restricted_exploitability']['expected_gain_matrix'],means)
                training=source['config']['training']
                assert training['trace'][-1]['profile']==training['profile']
                if training['status']=='converged':
                    assert training['trace'][-1]['changes']==0 and training['training_exploitability']<=1e-8
                else:
                    assert training['status']=='sweep_limit' and training['sweeps']==summary['budget']
        counts[exp]={'cells':len(summaries),'seed_rows':len(summaries)*config[exp]['n_seeds']}
    paired_checks=0
    for mode,metrics in read(rd/'e16'/'paired.json').items():
        for metric,stat in metrics.items():
            close(stat['mean_diff'],np.mean([a[metric]-b[metric] for a,b in zip(lookup[('e16',mode)],lookup[('e16','frozen')])]))
            paired_checks+=1
    for item in read(rd/'e17'/'paired.json'):
        for metric,stat in item['metrics'].items():
            a,b=lookup[('e17',item['rho'],'Karma-Cap')],lookup[('e17',item['rho'],'Karma')]
            close(stat['mean_diff'],np.mean([x[metric]-y[metric] for x,y in zip(a,b)]));paired_checks+=1
    for item in read(rd/'e18'/'paired.json'):
        a=lookup[('e18',item['label'],item['initial'],item['budget'])]
        b=lookup[('e18',item['label'],item['initial'],item['reference_budget'])]
        for metric,stat in item['metrics'].items():
            close(stat['mean_diff'],np.mean([x[metric]-y[metric] for x,y in zip(a,b)]));paired_checks+=1
    example_checks=0
    for label,data in read(rd/'e17'/'example.json').items():
        for name in ('dev','truthful'):
            trace=data[name]
            for row in trace:
                close(row['next_balance'],row['balance']-row['charge']+row['dividend']-row['ceiling_loss'])
                assert 0<=row['charge']<=row['balance']+1e-10
                if row['forced']:assert row['allocated']==1
                example_checks+=1
            totals=karma_statistics(trace)
            close(totals['credit_spent'],totals['forced_credit_spent']+totals['auction_credit_spent'])
    reproduction_checks=0
    c=config['e16']
    if (c['n'],c['T'],c['H'],c['n_rollouts'])==(10,120,40,80) and (rd/'e15'/'raw').exists():
        for mode in ('updating','frozen'):
            old=read(rd/'e15'/'raw'/f'Rank-Cap_{mode}_H40_R80.json')['rows']
            by_seed={r['seed']:r['M_rollout'] for r in old}
            for row in lookup[('e16',mode)]:
                close(row['M_rollout'],by_seed[row['seed']]);reproduction_checks+=1
    c=config['e17']
    if (c['n'],c['T'])==(50,1000) and (rd/'e12'/'raw').exists():
        for rho in (.25,1):
            for label in ('Karma','Karma-Cap'):
                old=read(rd/'e12'/'raw'/f"base_cap_2_rho{rho:g}_Karma_W{10 if label=='Karma-Cap' else None}.json")['rows']
                by_seed={r['seed']:r['M_uni'] for r in old}
                for row in lookup[('e17',rho,label)]:
                    close(row['M_uni'],by_seed[row['seed']]);reproduction_checks+=1
    result={'checks_passed':True,'studies':counts,'saved_mean_checks':mean_checks,'paired_metric_checks':paired_checks,
            'example_credit_transition_checks':example_checks,'original_seed_outcome_reproduction_checks':reproduction_checks,
            'rank_replay_note':'Private truth updates only the focal reference; fixed report sequence; endogenous counts and waits.',
            'seed_note':'E16/E17 reuse prior evaluation descriptively; E18 independently evaluates fixed training budgets.'}
    save_json(result,rd/'FOLLOWUP_AUDIT.json')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',default='results')
    audit(ap.parse_args().results_dir)
