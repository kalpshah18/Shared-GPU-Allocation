"""Generate scientific figures and the failure-mode report from E16-E18 outputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.course_plots import errorbar
from analysis.report import _f, _table
from analysis.failure_diagnostics import service_totals


def load(root,exp,name='summary.json'):
    source=Path(root)/exp/name
    return json.loads(source.read_text()) if source.exists() else None


def plot_rank_example(example):
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
    for column,mode in enumerate(('updating','frozen')):
        data=example[mode]
        attack,baseline,clean=data['attack'],data['baseline'],data['clean_reference']
        t=np.arange(len(attack))
        def field(rows,key):return np.array([r[key] for r in rows])
        axes[0,column].plot(t,field(attack,'value'),color='#555',alpha=.65,label='True value')
        axes[0,column].plot(t,field(attack,'report'),color='#2171b5',alpha=.8,label='Report')
        axes[0,column].set_ylabel('Value and report')
        axes[0,column].set_title(f"{mode} attacker; seed {data['seed']}")
        axes[0,column].legend(frameon=False,fontsize=8)
        axes[1,column].plot(t,field(attack,'true_value_rank'),label='Rank against reported history',alpha=.8)
        axes[1,column].plot(t,field(attack,'clean_true_value_rank'),label='Rank against truthful history',alpha=.8)
        axes[1,column].set_ylabel('Rank of the SAME true value')
        axes[1,column].legend(frameon=False,fontsize=8)
        for rows,label,color in ((baseline,'Truthful','#555'),(attack,'Attacker','#2171b5'),(clean,'Truthful-reference replay','#d95f0e')):
            axes[2,column].plot(t,np.cumsum(field(rows,'allocated')),label=label,color=color)
        forced=field(attack,'forced').astype(bool)
        axes[2,column].scatter(t[forced],np.cumsum(field(attack,'allocated'))[forced],marker='x',color='#2171b5',s=25,label='Forced attack service')
        axes[2,column].set_ylabel('Cumulative allocations')
        axes[2,column].legend(frameon=False,fontsize=8)
        for rows,label,color in ((attack,'Ordinary rank reference','#2171b5'),(clean,'Truthful-reference replay','#d95f0e')):
            gain=np.cumsum(field(rows,'utility')-field(baseline,'utility'))
            axes[3,column].plot(t,gain,label=label,color=color)
        axes[3,column].axhline(0,color='#666',linestyle='--',linewidth=.8)
        axes[3,column].set_ylabel('Cumulative utility gain')
        axes[3,column].set_xlabel('Round (zero indexed)')
        axes[3,column].legend(frameon=False,fontsize=8)
    fig.suptitle('E16: first locked workload; descriptive trajectory, not selected for attack success')
    fig.tight_layout()
    return fig


def plot_rank_summary(rows):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    x=np.arange(len(rows))
    for offset,key,label,color in ((-.22,'M_rollout','Actual gain','#2171b5'),
                                    (0,'clean_reference_gain','Truthful-reference replay','#d95f0e'),
                                    (.22,'ledger_effect','Actual minus replay','#238b45')):
        errorbar(axes[0],x+offset,rows,key,fmt='o',color=color,label=label)
    axes[0].set_ylabel('Episode utility difference (95% CI)')
    for offset,key,label,color in ((-.12,'underreport_fraction','Underreport','#2171b5'),
                                    (.12,'overreport_fraction','Overreport','#d95f0e')):
        errorbar(axes[1],x+offset,rows,key,fmt='o',color=color,label=label)
    axes[1].set_ylim(0,1)
    axes[1].set_ylabel('Fraction of rounds (95% CI)')
    for offset,key,label,color in ((-.12,'mean_history_rank_shift','All queries','#2171b5'),
                                    (.12,'mean_shift_high_values','True values >= 0.5','#d95f0e')):
        errorbar(axes[2],x+offset,rows,key,fmt='o',color=color,label=label)
    axes[2].set_ylabel('Rank shift from reported history (95% CI)')
    for ax in axes:
        ax.set_xticks(x,[r['mode'] for r in rows])
        ax.set_xlim(-.6,len(rows)-.4)
        ax.axhline(0,color='#666',linestyle='--',linewidth=.8)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('E16: utility replay, reporting choices and direct history effects across paired workloads')
    fig.tight_layout()
    return fig


def plot_karma(rows):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    x=np.arange(len(rows))
    panels=[(axes[0,0],('forced_gain','auction_gain'),('Forced service','Auction service'),'Unilateral benefit gain'),
            (axes[0,1],('forced_credit_spent_change','auction_credit_spent_change'),('Forced charges','Auction charges'),'Change in internal credit spent'),
            (axes[1,0],('deviating_balance_limited','truthful_balance_limited'),('Deviating focal','Truthful focal'),'Fraction of balance-limited bids'),
            (axes[1,1],('deviating_value_per_win','truthful_value_per_win'),('Deviating focal','Truthful focal'),'Mean true value per service')]
    for ax,keys,labels,ylabel in panels:
        for offset,key,label,color in zip((-.15,.15),keys,labels,('#2171b5','#d95f0e')):
            ax.bar(x+offset,[r[key]['mean'] for r in rows],width=.28,color=color,alpha=.7,label=label)
            errorbar(ax,x+offset,rows,key,fmt='none',ecolor=color)
        if ax==axes[0,0]:
            errorbar(ax,x,rows,'M_uni',fmt='D',color='#333',label='Total benefit gain')
        ax.axhline(0,color='#666',linestyle='--',linewidth=.8)
        ax.set_xticks(x,[f"rho={r['rho']:g}\n{r['label']}" for r in rows])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('E17: same opponents, focal truthful vs cap_2; credits are not monetary utility deductions')
    fig.tight_layout()
    return fig


def plot_adaptation(rows,root):
    starts=list(dict.fromkeys(r['initial'] for r in rows))
    labels=list(dict.fromkeys(r['label'] for r in rows))
    budgets=sorted({r['budget'] for r in rows})
    fig,axes=plt.subplots(3,len(starts),figsize=(12,11),squeeze=False)
    for col,initial in enumerate(starts):
        for index,label in enumerate(labels):
            source=load(root,'e18',f"raw/{label}_{initial}_budget{max(budgets)}.json")
            trace=source['config']['training']['trace']
            axes[0,col].plot([r['sweep'] for r in trace[1:]],[r['changes'] for r in trace[1:]],
                             marker='o',markersize=3,label=label,color=plt.cm.tab10(index))
        axes[0,col].set_title(f'Initial policy: {initial}')
        axes[0,col].set_xlabel('Training sweep')
        axes[0,col].set_ylabel('Users changing policy in sweep')
        axes[0,col].legend(frameon=False,fontsize=8,ncol=2)
        x=np.arange(len(labels))
        for index,budget in enumerate(budgets):
            selected=[next(r for r in rows if r['label']==label and r['initial']==initial and r['budget']==budget) for label in labels]
            offset=(index-(len(budgets)-1)/2)*.35
            axes[1,col].bar(x+offset,[r['training_exploitability'] for r in selected],width=.32,alpha=.75,label=f'Budget {budget}')
            axes[2,col].bar(x+offset,[r['restricted_exploitability']['mean'] for r in selected],width=.32,alpha=.75,label=f'Budget {budget}')
            errorbar(axes[2,col],x+offset,selected,'restricted_exploitability',fmt='none',ecolor='#333')
        for row,ylabel in ((1,'Training maximum unilateral gain'),(2,'Independent evaluation regret (simultaneous band)')):
            axes[row,col].set_xticks(x,labels,rotation=30,ha='right')
            axes[row,col].set_ylabel(ylabel)
            axes[row,col].set_ylim(bottom=0)
            axes[row,col].legend(frameon=False,fontsize=8)
    fig.suptitle('E18: extending training without choosing the budget from evaluation results')
    fig.tight_layout()
    return fig


def run_figures(results_dir='results',figures_dir='figures'):
    jobs=[('e16','e16_rank_trajectory',lambda rows:plot_rank_example(load(results_dir,'e16','example.json'))),
          ('e16','e16_rank_channels',plot_rank_summary),('e17','e17_karma_accounting',plot_karma),
          ('e18','e18_extended_adaptation',lambda rows:plot_adaptation(rows,results_dir))]
    dest=Path(figures_dir)
    for exp,stem,build in jobs:
        rows=load(results_dir,exp)
        if rows is None:continue
        fig=build(rows)
        dest.mkdir(parents=True,exist_ok=True)
        for suffix in ('png','pdf'):
            fig.savefig(dest/f'{stem}.{suffix}',bbox_inches='tight',dpi=160)
        plt.close(fig)
        print(f'Saved {stem}.{{png,pdf}}')


def build_report(results_dir='results'):
    rank,karma,adaptive=[load(results_dir,exp) for exp in ('e16','e17','e18')]
    if any(rows is None for rows in (rank,karma,adaptive)):
        raise FileNotFoundError('Generate E16, E17 and E18 results before writing the combined report.')
    configs={exp:load(results_dir,exp,'config.json') for exp in ('e16','e17','e18')}
    paired=load(results_dir,'e16','paired.json')['updating']
    def difference(key):
        stat=paired[key]
        return _f({key:{**stat,'mean':stat['mean_diff']}},key,3,ci=True,signed=True)
    parts=['# Why the allocation mechanisms fail\n',
           'Follow-up to the component study: report-history dynamics, internal credit accounting, and longer adaptive training.\n',
           '## Design and interpretation\n',
           f"E16 and E17 reuse {configs['e16']['n_seeds']} existing evaluation workloads to explain previous results. "
           f"E18 uses {configs['e18']['n_seeds']} newly locked independent evaluation seeds and trains only on seeds "
           f"{configs['e18']['training_seeds']}. The [fixed follow-up protocol](FOLLOWUP_PROTOCOL.md) specifies budgets "
           'and comparisons before these runs. No mechanism tuning uses these results. Per-seed bootstrap intervals '
           'are 95%; full runs use 10,000 resamples. Mechanistic identities, replay interventions, and empirical gains '
           'answer different questions.\n',
           '## 1. Rank-Cap: modelling the cost of changing report history\n',
           f"The two attackers use n={configs['e16']['n']}, k={configs['e16']['k']}, T={configs['e16']['T']}, "
           f"H={configs['e16']['H']}, and {configs['e16']['n_rollouts']} simulated futures per candidate. "
           'Both actual mechanisms update real histories. The frozen control freezes only hypothetical histories '
           'inside the planner. Workloads, ties, and attacker random streams are paired.\n']
    parts.append(_table(['Attacker','Actual gain [CI]','Forced gain','Auction gain','Truthful-reference replay gain [CI]','Actual minus replay [CI]'],
                        [[r['mode'],_f(r,'M_rollout',2,ci=True,signed=True),_f(r,'forced_gain',2,signed=True),
                          _f(r,'auction_gain',2,signed=True),_f(r,'clean_reference_gain',2,ci=True,signed=True),
                          _f(r,'ledger_effect',2,ci=True,signed=True)] for r in rank]))
    parts.append('Updating-minus-frozen utility gain is '+difference('M_rollout')+'. This isolates the planner\'s '
                 'history approximation, not the causal utility contribution of the report ledger.\n')
    parts.append(_table(['Attacker','Underreport fraction','Overreport fraction','Mean true value when understating',
                         'Rank shift: all queries [CI]','Rank shift: high-value queries [CI]'],
                        [[r['mode'],_f(r,'underreport_fraction',3),_f(r,'overreport_fraction',3),_f(r,'underreport_value',3),
                          _f(r,'mean_history_rank_shift',3,ci=True,signed=True),_f(r,'mean_shift_high_values',3,ci=True,signed=True)] for r in rank]))
    parts.append('**The direct history effect.** For a fixed current query z, replace one past truthful report v by r '
                 'in a ledger of size m. The empirical rank changes by\n\n'
                 '```text\n([r<z] + 0.5[r=z] - [v<z] - 0.5[v=z]) / (m+1).\n```\n\n'
                 'Understatement raises the future rank for r<z<v; overstatement lowers it for v<z<r. '
                 'This compares the same query and ledger size. The logged shift compares the current TRUE value '
                 'against reported and truthful past values, so it does not confuse a changed current report with '
                 'a changed reference distribution. Allocation counts and waits introduce additional feedback.\n')
    updating=next(r for r in rank if r['mode']=='updating')
    frozen=next(r for r in rank if r['mode']=='frozen')
    if updating['overreport_fraction']['mean'] < frozen['overreport_fraction']['mean']:
        parts.append('The updating planner exaggerates less often than the frozen planner. Understatements also occur '
                     'on relatively high values, so the trace does not support a simple rule of sacrificing only '
                     'low-value rounds to build a favourable reference. A planner can lower an already sufficient '
                     'winning report and thereby limit the later rank penalty of earlier exaggeration. The per-seed '
                     'paired rank-shift and reporting contrasts are saved alongside the utility comparison.\n')
    if updating['ledger_effect']['ci_lower'] > 0:
        parts.append('**Report history contributes to the gain, but is not the whole explanation.** Replacing '
                     'the attacked ledger with truthful reference values reduces the updating attacker\'s '
                     'fixed-sequence utility gain, with a paired interval above zero for actual-minus-replay. '
                     'This intervention lets allocation counters and waits evolve, so its effect includes '
                     'feedback from changed service. The measured rank shift for high-value queries '
                     'shows that selective reporting can improve the reference at valuable future queries '
                     'even when its overall average rank shift is close to zero. The frozen planner instead '
                     'exaggerates more often and suffers a negative mean reference shift.\n')
        if updating['clean_reference_gain']['ci_lower'] > 0:
            parts.append('The updating attack is still profitable under truthful-reference replay. Its benefit '
                         'therefore includes current reporting/allocation effects beyond changes to the ledger. '
                         'The paired ledger effect is not an additive decomposition of the optimal adaptive '
                         'attack\'s incentives: reports are held fixed, and the attacker is not reoptimised. '
                         'Truthful-reference replay uses private truth and is a diagnostic, not a feasible '
                         'strategy-proof replacement mechanism.\n')
    elif updating['ledger_effect']['ci_upper'] < 0:
        parts.append('**The replay rules out a simple poisoning explanation.** Replaying the SAME reports with a '
                     'truthful focal reference earns more utility than ordinary replay on average, with a paired '
                     'interval excluding zero. The observed attack does not require contaminated history to be '
                     'profitable: contamination remains a cost that the updating planner better accounts for. '
                     'This privileged replay uses private truth, and the attack sequence is not reoptimised against it; '
                     'it is a diagnostic intervention, not a proposed strategy-proof mechanism.\n')
    else:
        parts.append('The truthful-reference replay measures the utility effect of removing report-history changes '
                     'for a FIXED attack sequence, while counts and waits evolve. Its interval must be read before '
                     'claiming that contamination helps or hurts. It uses private truth and does not reoptimise the '
                     'attacker, so it is a diagnostic intervention rather than a feasible mechanism or equilibrium test.\n')
    parts.append(f"The descriptive trajectory uses the first locked seed, {configs['e16']['example_seed']}, "
                 'rather than the most profitable attack. All seed trajectories are saved. '
                 '[Trajectory figure](../figures/e16_rank_trajectory.pdf); [aggregate channels](../figures/e16_rank_channels.pdf).\n')
    example=load(results_dir,'e16','example.json')['updating']
    effect=service_totals(example['attack'])['utility']-service_totals(example['clean_reference'])['utility']
    parts.append(f"The first-seed ledger effect is {effect:+.2f}, whereas the aggregate estimate is "
                 + _f(updating,'ledger_effect',2,ci=True,signed=True)
                 + '. The example illustrates variation and should not be treated as representative of the mean effect.\n')
    parts.append('## 2. Karma: service channels and the price of a deviation\n')
    parts.append(f"E17 uses n={configs['e17']['n']}, k={configs['e17']['k']}, T={configs['e17']['T']} and the "
                 'original focal users. For each pair, only the focal changes between truthful and cap_2 reporting; '
                 'all opponents keep their policies. The benefit identity is exact:\n\n'
                 '```text\nGain = change in forced-service benefit + change in auction-service benefit.\n```\n')
    parts.append(_table(['Rule','rho','Total gain [CI]','Forced benefit gain [CI]','Auction benefit gain [CI]',
                         'Change in services','Change in credits spent'],
                        [[r['label'],str(r['rho']),_f(r,'M_uni',2,ci=True,signed=True),_f(r,'forced_gain',2,ci=True,signed=True),
                          _f(r,'auction_gain',2,ci=True,signed=True),_f(r,'wins_change',2,signed=True),
                          _f(r,'credit_spent_change',2,signed=True)] for r in karma]))
    parts.append(_table(['Rule','rho','Deviator bids limited by balance','Truthful bids limited by balance',
                         'Deviator value/service','Truthful value/service'],
                        [[r['label'],str(r['rho']),_f(r,'deviating_balance_limited',3),_f(r,'truthful_balance_limited',3),
                          _f(r,'deviating_value_per_win',3),_f(r,'truthful_value_per_win',3)] for r in karma]))
    parts.append(_table(['Rule','rho','Deviator mean balance','Truthful mean balance','Forced credit change','Auction credit change'],
                        [[r['label'],str(r['rho']),_f(r,'deviating_mean_balance',3),_f(r,'truthful_mean_balance',3),
                          _f(r,'forced_credit_spent_change',2,signed=True),_f(r,'auction_credit_spent_change',2,signed=True)] for r in karma]))
    parts.append('**Forced service is charged.** An auction winner pays the remaining auction\'s uniform price; a '
                 'forced winner pays min(overall slot price, its balance). Credits are redistributed equally, then '
                 'balances are clipped at the ceiling. Every observed transition is checked against the real rule. '
                 'Karma is not deducted from utility: a large credit payment affects later bids, not current net benefit.\n')
    full_off=next(r for r in karma if r['rho']==1 and r['label']=='Karma')
    full_on=next(r for r in karma if r['rho']==1 and r['label']=='Karma-Cap')
    if full_off['M_uni']['ci_lower']>0 and full_on['M_uni']['ci_lower']>0:
        parts.append('**The cap is not necessary for the population reversal.** When everyone else exaggerates, '
                     'positive mean deviation gains occur even with the cap disabled. Comparing the exact benefit '
                     'channels distinguishes a cap contribution from the auction\'s strategic competition. Higher '
                     'reports also encounter balance caps; price, credit recycling, service count and value per win '
                     'must be considered together. The accounting describes the reversal but does not prove a '
                     'unique equilibrium or identify one universally sufficient cause.\n')
        parts.append('At rho=0.25 the deviator obtains nearly the same number of services but a lower true '
                     'value per service than its truthful counterpart. At rho=1, exaggeration instead obtains '
                     'substantially more services; with the cap, the two value-per-service averages are similar. '
                     'The capped gain comes from additional auction benefit while forced-service benefit falls. '
                     'This supports a competition/quantity explanation rather than a benefit from uncharged '
                     'forced service. The deviator spends more credits, and those credits are a future bidding '
                     'constraint rather than an immediate utility cost. These are population-dependent '
                     'diagnostics on the tested cap_2 policy, not a general characterisation of karma games.\n')
        parts.append('For Karma-Cap the deviator\'s balance limits substantially fewer bids at rho=1 than at '
                     'rho=0.25, and its mean balance is higher. Thus the opportunity-cost deterrent seen against '
                     'mostly truthful opponents is weaker in the inflated-bid population. This observed budget '
                     'slack accompanies the service-count reversal; it is not a claim that changing the initial '
                     'credit supply alone would reproduce all of the population effect.\n')
    parts.append('[Accounting figure](../figures/e17_karma_accounting.pdf).\n')
    parts.append('## 3. Does a longer adaptive training budget resolve residual incentives?\n')
    parts.append('The four- and twenty-sweep profiles are trained independently of evaluation. A repeated '
                 'sweep-boundary profile is recorded but does not stop the longer search; shuffled future orders '
                 'may behave differently. Converged means that an entire sweep has no profitable policy changes '
                 'on the training workload game.\n')
    for initial in dict.fromkeys(r['initial'] for r in adaptive):
        selected=[r for r in adaptive if r['initial']==initial]
        parts.append(f'**Initial policy: {initial}.**\n')
        parts.append(_table(['Rule','Budget','Status','Sweeps used','Recurrences','Training regret','Independent regret [band]','WR [CI]'],
                            [[r['label'],str(r['budget']),r['training_status'],str(r['training_sweeps']),str(len(r['recurrent_sweeps'])),
                              f"{r['training_exploitability']:.3f}",_f(r,'restricted_exploitability',3,ci=True),
                              _f(r,'WR',3,ci=True)] for r in selected]))
    longest=max(r['budget'] for r in adaptive)
    unresolved=sorted({r['label'] for r in adaptive if r['budget']==longest and r['training_status']!='converged'})
    if unresolved:
        parts.append(f"At budget {longest}, {', '.join(unresolved)} still have residual training incentives. "
                     'Longer training therefore does not establish a fixed point for every mechanism.\n')
    karma_long=[r for r in adaptive if r['label']=='Karma-Cap' and r['budget']==longest]
    if karma_long and all(r['training_status']=='converged' for r in karma_long):
        parts.append('Karma-Cap reaches a training fixed point from both starting profiles after '
                     f"{max(r['training_sweeps'] for r in karma_long)} sweeps. Its longer-trained welfare and "
                     'independent regret must still be read from the table; convergence refers only to the training game.\n')
    positive=[r for r in adaptive if r['restricted_exploitability']['ci_lower']>0]
    if positive:
        parts.append('Independent regret bands exclude zero for '+', '.join(
                     f"{r['label']} from {r['initial']} at budget {r['budget']}" for r in positive)
                     + '. Thus the earlier zero-containing bands do not establish robust equilibrium behaviour '
                     'on a new evaluation set. The fixed budgets were not selected using these results.\n')
    parts.append('A training fixed point is a restricted pure-policy equilibrium on five training traces, not '
                 'unrestricted dynamic strategyproofness. An evaluation interval containing zero does not establish '
                 'positive expected regret; a sweep limit or recurrence with residual training regret prevents a '
                 'training equilibrium claim. [Training-budget figure](../figures/e18_extended_adaptation.pdf).\n')
    parts.append('## Course presentation outline\n\n'
                 '1. State the repeated private-value allocation model and the welfare/service/incentive question.\n'
                 '2. Show Score-Cap as the matched service baseline and the formal waiting bound.\n'
                 '3. Show the corrected Rank-Cap attack, the first-seed trace, and truthful-reference replay.\n'
                 '4. Explain Karma\'s reversal using forced/auction benefit and credit accounting.\n'
                 '5. Compare four-vs-twenty training sweeps on independent evaluation seeds.\n'
                 '6. Close with the constrained welfare benchmark and the precise limits of each claim.\n')
    parts.append('## Reproduction\n\n'
                 '```text\npython run_all.py --followups-only\n```\n\n'
                 'This runs E0 and E16-E18, then generates the figures and this report. Use `--quick` and separate '
                 'results/figure directories for development. Older experiment outputs are retained.\n')
    if load(results_dir,'e18','config.json')['n_seeds']==30:
        parts.append('The [follow-up audit](../results/FOLLOWUP_AUDIT.json) checks independent evaluation seeds, '
                     'saved means, paired differences, credit transitions and original seed-level reproduction.\n')
    return '\n\n'.join(parts)+'\n'


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',default='results')
    ap.add_argument('--figures-dir',default='figures')
    ap.add_argument('--out',default='docs/FAILURE_CASE_STUDIES.md')
    args=ap.parse_args()
    run_figures(args.results_dir,args.figures_dir)
    destination=Path(args.out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(build_report(args.results_dir),encoding='utf-8')
    print(f'Wrote {destination}')
