# Formal properties of the service override and pure Rank-Cap

## 1. Waiting guarantee, including overflow

Assume n users, exactly k<n services per round, zero initial waits, integer W>=1,
and an oldest-first override: when q_i(t)>=W, up to k eligible users with largest
waits must be served before ordinary priorities. Let C=ceil(n/k).

**Proposition.** For every report sequence, q_i(t)<=W+C-1 for every user and
round, including the terminal state after any finite episode.

**Proof.** Consider a user i at the first round t at which its wait reaches W.
At most n-1 users can precede it in oldest-first order. While i remains unserved,
its wait increases each round. A preceding user that is served resets its wait
to zero and cannot become older than i again before i is served. A previously
younger user cannot overtake i because both waits increase at the same rate.
Whenever i is not served, all k slots go to preceding eligible users: otherwise
the override would select i. Thus at most floor((n-1)/k)=C-1 such rounds pass
before i is served. Its largest pre-service wait is W+C-1. Before eligibility its
wait is below W, and after service it resets to zero. Apply this argument to
every waiting interval. QED.

The threshold W is NOT an unconditional bound of W. If more than k users become
eligible together, excess users wait longer. A claimed Q_max=W from a simulation
is an observation, not the worst-case guarantee. Zero SR_Delta is likewise not
guaranteed merely by choosing W=Delta: overflow may exceed Delta.

**Necessary capacity condition.** A uniform bound B over every sufficiently long
episode requires k(B+1)>=n. Otherwise a window of B+1 rounds has too few service
slots to serve every user. Round-robin attains B=ceil(n/k)-1.

## 2. Exact finite-episode constrained oracle

With q_i(0)=0, integer B>=0 and binary allocation x, q_i(t)<=B for all
t=0,...,T iff every length B+1 window in rounds 0,...,T-1 contains at least one
service for i. If T<=B there are no such windows and the condition is automatic.

If a user receives no service in a full window, its wait immediately afterwards
is at least B+1. Conversely, a wait exceeding B implies that the previous B+1
rounds were unserved. This includes the first and last possible windows.

The binary optimisation is therefore:

    maximise sum_{i,t} v[i,t] * x[i,t]
    subject to sum_i x[i,t] = k                         for every t
               sum_{t=s}^{s+B} x[i,t] >= 1             for every i, 0<=s<T-B
               x[i,t] in {0,1}.

Let W* be unconstrained welfare, W*_B constrained optimum, and W_A the welfare
of an online schedule satisfying the same B. Then W_A<=W*_B<=W* and

    (W* - W_A)/W* = (W* - W*_B)/W* + (W*_B - W_A)/W*.

The first term is the unavoidable service cost on that workload. The second is
the remaining algorithm loss relative to a future-aware benchmark; it includes
the online information disadvantage. It is not an assertion that an online
algorithm can attain W*_B. Solver bounds replace W*_B when not certified optimal.

## 3. Rank invariance under fixed strictly increasing distortions

Assume pure rank (rank_blend=0), identical initial mechanism/allocation histories,
identical tie-breaking, and for each user a fixed strictly increasing function
f_i mapping the report domain into the permitted report interval. Transform
EVERY report of user i, including its initial report history, by f_i.

**Proposition.** Every allocation remains identical, with or without the wait cap.

**Proof.** Strict increase preserves every '<' and '=' comparison between the
current report and that user's past reports. Hence the empirical quantile

    (#past below + 0.5*#past equal + 0.5)/(#past+1)

is unchanged. Initially cumulative allocations and waits are equal, so rank
scores, forced users and seeded decisions are equal. After a common allocation,
allocation counters remain equal, and the histories still correspond under f_i.
Induction on rounds gives identical allocation paths. QED.

This is ordinal invariance, NOT strategyproofness. Clipping is not strictly
increasing; time-varying distortions do not preserve comparisons with earlier
reports; selective history manipulation is outside the theorem. A nonzero raw
report blend also breaks the premise. The theorem concerns allocations; applying
a transform to true values changes welfare and utility, so those are not invariant.

## 4. Relation to published Karma

Karma-Cap is this project's internal uniform-price bid auction with redistribution
and a service override. Published **Karma (OSDI 2023)** instead assigns credits
for donating unused resources that others borrow in a dynamic-demand allocation
model. Its online strategy-proofness and efficiency results do not transfer to
our auction. Here both truthfulness and equilibrium behaviour are empirical.

Source: https://www.usenix.org/conference/osdi23/presentation/vuppalapati
