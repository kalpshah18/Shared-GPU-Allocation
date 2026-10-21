"""Score with the same oldest-first service override as M6 and M7."""
import numpy as np

from sim.mechanisms.m4_score import ScoreMechanism
from sim.mechanisms._utils import batch_top_k_mask
from sim.mechanisms._waitcap import forced_mask, with_forced_priority


class ScoreCapMechanism(ScoreMechanism):
    def _step(self, reports, cumulative, wait, tiebreak):
        scores = self.scores(reports, cumulative)
        forced = (forced_mask(wait, self.cfg.wait_limit, self.cfg.k, tiebreak)
                  if self.cfg.enable_wait_cap else np.zeros(reports.shape, dtype=np.int64))
        x = batch_top_k_mask(with_forced_priority(scores, forced), tiebreak, self.cfg.k)
        return x, np.zeros(reports.shape, dtype=np.float64)

    def allocate(self, reports, history, tie_seed):
        tb = np.random.default_rng(tie_seed).random(self.cfg.n)[None, :]
        x, p = self._step(reports[None, :], history.cumulative[None, :],
                          history.consecutive_wait[None, :], tb)
        return x[0], p[0]

    def allocate_batch(self, reports, cumulative, tiebreak):
        raise NotImplementedError("Score-Cap needs waiting state; use rollout_step.")

    def rollout_step(self, reports, state, tiebreak):
        x, p = self._step(reports, state["cum"], state["wait"], tiebreak)
        state["cum"] = state["cum"] + x
        state["wait"] = np.where(x == 1, 0, state["wait"] + 1)
        return x, p
