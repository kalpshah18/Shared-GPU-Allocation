"""Finite-policy repeated game; choices are trained on independent workload seeds.

Users choose a policy for a whole allocation episode. Updates are asynchronous
best responses to expected episode utility on training traces, not clairvoyant
per-evaluation-trace decisions. Convergence is only within this policy library.
"""
import numpy as np

from sim.environment import History, SeedPackage
from sim.mechanisms import make_mechanism
from sim.metrics import utilities
from sim.policies import call_policy, get_policy
from sim.runner import _checked

POLICY_NAMES = ("truthful", "cap_1.25", "cap_1.5", "cap_2", "max_claim",
                "timed_cap_1.25", "timed_cap_2", "high_only")


def high_only(v, history, cfg):
    return np.where(np.asarray(v) >= 0.5 * cfg.v_max, v, 0.0)


def run_profile(mname, pkg, cfg, profile, names=POLICY_NAMES):
    if len(profile) != cfg.n or any(p < 0 or p >= len(names) for p in profile):
        raise ValueError("One valid policy index is required per user.")
    groups = []
    for p in sorted(set(profile)):
        users = np.flatnonzero(np.asarray(profile) == p)
        policy = high_only if names[p] == "high_only" else get_policy(names[p])
        groups.append((users, policy))
    mech, history = make_mechanism(mname, cfg, pkg), History(cfg.n)
    for t in range(cfg.T):
        values, reports = pkg.valuations[:, t], pkg.valuations[:, t].copy()
        for users, policy in groups:
            reports[users] = call_policy(policy, values[users], history, cfg, users=users)
        x, p = mech.allocate(_checked(reports, cfg), history, int(pkg.tie_seeds[t]))
        history.update(x, p)
    return history


class ProfileEvaluator:
    """Cache expected utilities, using identical traces for all policy profiles."""
    def __init__(self, mname, cfg, seeds, names=POLICY_NAMES):
        if not seeds:
            raise ValueError("At least one training seed is required.")
        self.mname, self.cfg, self.names = mname, cfg, tuple(names)
        self.packages = [SeedPackage.generate(seed, cfg) for seed in seeds]
        self.cache = {}

    def utility(self, profile):
        profile = tuple(profile)
        if profile not in self.cache:
            self.cache[profile] = np.mean([utilities(run_profile(self.mname, pkg, self.cfg, profile, self.names),
                                                       pkg.valuations) for pkg in self.packages], axis=0)
        return self.cache[profile]

    def gains(self, profile):
        base = self.utility(profile)
        gains = np.zeros((self.cfg.n, len(self.names)))
        for user in range(self.cfg.n):
            for policy in range(len(self.names)):
                changed = list(profile)
                changed[user] = policy
                gains[user, policy] = self.utility(changed)[user] - base[user]
        return gains


def best_response_dynamics(evaluator, initial, max_sweeps=4, order_seed=329, tolerance=1e-8,
                           stop_on_recurrence=True):
    """Keep the current policy on ties. Detect recurrence after each full sweep."""
    if max_sweeps < 1 or tolerance < 0:
        raise ValueError("Positive sweeps and non-negative tolerance are required.")
    profile = list(initial)
    trace = [dict(sweep=0, profile=list(profile), expected_utility=evaluator.utility(profile).tolist())]
    seen = {tuple(profile): 0}
    rng, status = np.random.default_rng(order_seed), "sweep_limit"
    recurrent_sweeps = []
    for sweep in range(1, max_sweeps + 1):
        changes = 0
        order = rng.permutation(evaluator.cfg.n).tolist()
        for user in order:
            current = evaluator.utility(profile)[user]
            options = []
            for policy in range(len(evaluator.names)):
                changed = list(profile)
                changed[user] = policy
                options.append(evaluator.utility(changed)[user])
            best = int(np.argmax(options))
            if options[best] > current + tolerance:
                profile[user] = best
                changes += 1
        trace.append(dict(sweep=sweep, profile=list(profile), changes=changes, update_order=order,
                          expected_utility=evaluator.utility(profile).tolist()))
        if changes == 0:
            status = "converged"
            break
        if tuple(profile) in seen:
            recurrent_sweeps.append(sweep)
            if stop_on_recurrence:
                status = "cycle"
                break
        seen[tuple(profile)] = sweep
    gains = evaluator.gains(profile)
    return {"profile": profile, "status": status, "sweeps": sweep, "trace": trace,
            "recurrent_sweeps": recurrent_sweeps,
            "training_exploitability": float(gains.max()), "profiles_evaluated": len(evaluator.cache)}
