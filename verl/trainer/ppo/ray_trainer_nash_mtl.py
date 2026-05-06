"""
Nash-MTL PPO trainer with Ray backend.

This trainer reuses the MGDA multi-objective data path: rewards are converted
to per-objective advantages on the driver, while the actor worker computes
per-objective policy gradients and combines them with Nash bargaining weights.
"""

from verl.trainer.ppo.ray_trainer_mgda import MGDARayPPOTrainer


class _NashMTLActorRolloutProxy:
    """Delegate all worker-group calls, remapping the MGDA update hook to Nash-MTL."""

    def __init__(self, worker_group):
        self._worker_group = worker_group

    def __getattr__(self, name):
        return getattr(self._worker_group, name)

    def update_actor_mgda(self, *args, **kwargs):
        return self._worker_group.update_actor_nash_mtl(*args, **kwargs)


class NashMTLRayPPOTrainer(MGDARayPPOTrainer):
    """PPO trainer for the Nash-MTL multi-objective baseline."""

    def init_workers(self):
        super().init_workers()
        self.actor_rollout_wg = _NashMTLActorRolloutProxy(self.actor_rollout_wg)
