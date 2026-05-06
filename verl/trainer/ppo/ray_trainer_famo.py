"""
FAMO PPO trainer with Ray backend.

The driver reuses the MGDA path to build per-objective advantages, while the
actor worker applies the FAMO weighted loss and updates FAMO task logits after
the actor step on the same rollout batch.
"""

from verl.trainer.ppo.ray_trainer_mgda import MGDARayPPOTrainer


class _FAMOActorRolloutProxy:
    """Delegate all worker-group calls, remapping the MGDA update hook to FAMO."""

    def __init__(self, worker_group):
        self._worker_group = worker_group

    def __getattr__(self, name):
        return getattr(self._worker_group, name)

    def update_actor_mgda(self, *args, **kwargs):
        return self._worker_group.update_actor_famo(*args, **kwargs)


class FAMORayPPOTrainer(MGDARayPPOTrainer):
    """PPO trainer for the FAMO multi-objective baseline."""

    def init_workers(self):
        super().init_workers()
        self.actor_rollout_wg = _FAMOActorRolloutProxy(self.actor_rollout_wg)
