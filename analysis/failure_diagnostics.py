"""Observers and fixed-sequence interventions for mechanism failure case studies.

Observers call the real mechanism. Truthful rank-reference replay is an explicit
privileged-information intervention; it is not a proposed online mechanism.
"""
import bisect
import numpy as np

from sim.environment import History
from sim.mechanisms import KarmaCapMechanism, RankCapMechanism
from sim.mechanisms._waitcap import forced_mask
from sim.policies import rollout_report


def rank_trace(pkg, cfg, reports=None, H=40, R=80, truth_reference=False, focal=0):
    mechanism, history = RankCapMechanism(cfg), History(cfg.n)
    rng = np.random.default_rng([pkg.master_seed, 159, H, R])
    true_past, trace = [], []
    for t in range(cfg.T):
        values = pkg.valuations[:, t]
        submitted = values.copy()
        submitted[focal] = (reports[t] if reports is not None else
                            rollout_report(float(values[focal]), history, mechanism, cfg, focal,
                                           lambda v, h, c: v, rng, H=H, n_rollouts=R))
        own_past = mechanism._past[focal]
        past_mean = float(np.mean(own_past)) if own_past else 0.0
        rank = mechanism._quantile(own_past, float(submitted[focal]))
        truthful_query_rank = mechanism._quantile(own_past, float(values[focal]))
        clean_query_rank = mechanism._quantile(true_past, float(values[focal]))
        tb = np.random.default_rng(int(pkg.tie_seeds[t])).random(cfg.n)[None, :]
        forced = (forced_mask(history.consecutive_wait[None, :], cfg.wait_limit, cfg.k, tb)[0]
                  if cfg.enable_wait_cap else np.zeros(cfg.n, dtype=int))
        before = int(history.cumulative[focal])
        wait = int(history.consecutive_wait[focal])
        x, p = mechanism.allocate(submitted, history, int(pkg.tie_seeds[t]))
        trace.append(dict(round=t, value=float(values[focal]), report=float(submitted[focal]),
                          rank=float(rank), true_value_rank=float(truthful_query_rank),
                          clean_true_value_rank=float(clean_query_rank),
                          history_rank_shift=float(truthful_query_rank-clean_query_rank),
                          prior_allocations=before, wait=wait, forced=int(forced[focal]),
                          score=float(rank/(1+before)**cfg.lambda_), allocated=int(x[focal]),
                          utility=float(values[focal]*x[focal]), history_report_mean=past_mean))
        # The ledger has just appended the current report. Remove that exact
        # entry before inserting truth in the privileged replay intervention.
        if truth_reference:
            own_past.pop(bisect.bisect_left(own_past, float(submitted[focal])))
            bisect.insort(own_past, float(values[focal]))
        bisect.insort(true_past, float(values[focal]))
        history.update(x, p)
    return history, trace


def rank_statistics(trace):
    v = np.array([r["value"] for r in trace])
    r = np.array([r["report"] for r in trace])
    shift = np.array([row["history_rank_shift"] for row in trace])
    return dict(underreport_fraction=float(np.mean(r < v-1e-12)),
                overreport_fraction=float(np.mean(r > v+1e-12)),
                mean_report=float(r.mean()), mean_value=float(v.mean()),
                mean_history_rank_shift=float(shift.mean()),
                history_boost_fraction=float(np.mean(shift > 1e-12)),
                mean_shift_high_values=float(shift[v >= .5].mean()) if np.any(v >= .5) else 0.0,
                underreport_value=float(v[r < v-1e-12].mean()) if np.any(r < v-1e-12) else 0.0)


def service_totals(trace):
    utility = np.array([r["utility"] for r in trace])
    forced = np.array([r["forced"] for r in trace], dtype=bool)
    wins = np.array([r["allocated"] for r in trace])
    return dict(utility=float(utility.sum()), forced_utility=float(utility[forced].sum()),
                auction_utility=float(utility[~forced].sum()), wins=int(wins.sum()),
                forced_wins=int(wins[forced].sum()), auction_wins=int(wins[~forced].sum()))


def karma_trace(pkg, cfg, strategic_set, multiplier=2.0, observe=None):
    """Record internal charges and verify their transition against the real rule."""
    observe = np.arange(cfg.n) if observe is None else np.asarray(observe, dtype=int)
    mechanism, history = KarmaCapMechanism(cfg), History(cfg.n)
    traces = {int(i): [] for i in observe}
    strategic_set = np.asarray(strategic_set, dtype=int)
    for t in range(cfg.T):
        values = pkg.valuations[:, t]
        reports = values.copy()
        reports[strategic_set] = np.minimum(multiplier*values[strategic_set], cfg.v_max)
        balance = mechanism.balance.copy()
        bids = np.minimum(reports, balance)
        tb = np.random.default_rng(int(pkg.tie_seeds[t])).random(cfg.n)[None, :]
        forced = (forced_mask(history.consecutive_wait[None, :], cfg.wait_limit, cfg.k, tb)[0]
                  if cfg.enable_wait_cap else np.zeros(cfg.n, dtype=int))
        available = cfg.k-int(forced.sum())
        remaining = np.sort(bids[forced == 0])[::-1]
        price = float(remaining[available]) if available else 0.0
        slot_price = float(np.sort(bids)[::-1][cfg.k])
        waits = history.consecutive_wait.copy()
        x, p = mechanism.allocate(reports, history, int(pkg.tie_seeds[t]))
        charges = np.where(forced == 1, np.minimum(slot_price, balance),
                           x*np.minimum(price, balance))
        dividend = float(charges.sum()/cfg.n)
        unclipped = balance-charges+dividend
        ceiling_loss = np.maximum(unclipped-mechanism.balance_ceiling, 0)
        np.testing.assert_allclose(mechanism.balance, np.clip(unclipped, 0, mechanism.balance_ceiling), atol=1e-12)
        for i in observe:
            traces[int(i)].append(dict(round=t, value=float(values[i]), report=float(reports[i]),
                                      bid=float(bids[i]), balance=float(balance[i]), wait=int(waits[i]),
                                      forced=int(forced[i]), allocated=int(x[i]),
                                      utility=float(values[i]*x[i]), charge=float(charges[i]),
                                      auction_price=price, slot_price=slot_price, dividend=dividend,
                                      next_balance=float(mechanism.balance[i]),
                                      ceiling_loss=float(ceiling_loss[i]),
                                      balance_limited=int(balance[i] < reports[i]-1e-12)))
        history.update(x, p)
    return history, traces


def karma_statistics(trace):
    totals = service_totals(trace)
    totals.update(credit_spent=float(sum(r["charge"] for r in trace)),
                  forced_credit_spent=float(sum(r["charge"]*r["forced"] for r in trace)),
                  auction_credit_spent=float(sum(r["charge"]*(1-r["forced"]) for r in trace)),
                  balance_limited_fraction=float(np.mean([r["balance_limited"] for r in trace])),
                  mean_balance=float(np.mean([r["balance"] for r in trace])),
                  mean_bid=float(np.mean([r["bid"] for r in trace])),
                  value_per_win=totals["utility"]/totals["wins"] if totals["wins"] else 0.0)
    return totals
