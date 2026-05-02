"""
Smooth Tchebycheff PPO trainer with online mirror descent.
"""

from __future__ import annotations

from typing import Optional

import torch

import verl.trainer.ppo.ray_trainer_tchebycheff as base_tchebycheff
from verl.trainer.ppo.ray_trainer_tchebycheff import TchebycheffRayPPOTrainer


class SmoothTchebycheffRayPPOTrainer(TchebycheffRayPPOTrainer):
    """Tchebycheff trainer variant with smooth weights and OMD updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self.config.algorithm.get("smooth_tchebycheff", {})
        self._smooth_tchebycheff_omd_lr = cfg.get("omd_lr", 1.0)
        self._smooth_tchebycheff_temperature = cfg.get("temperature", 1.0)
        self._smooth_tchebycheff_eps = cfg.get("eps", 1e-8)
        self._smooth_tchebycheff_init = cfg.get("init", "uniform")

        self._preference_weights = self._scalarization_weights.clone()
        self._smooth_tchebycheff_log_weights: Optional[torch.Tensor] = None
        self._last_smooth_tchebycheff_weights: Optional[torch.Tensor] = None
        self._last_smooth_tchebycheff_objective_values: Optional[torch.Tensor] = None
        self._smooth_tchebycheff_allow_state_updates = True

    def _init_log_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._smooth_tchebycheff_init == "uniform":
            init_weights = torch.full(
                (self._preference_weights.numel(),),
                1.0 / self._preference_weights.numel(),
                device=device,
                dtype=dtype,
            )
        else:
            init_weights = self._preference_weights.to(device=device, dtype=dtype)
            init_weights = init_weights / init_weights.sum().clamp_min(self._smooth_tchebycheff_eps)

        return torch.log(init_weights.clamp_min(self._smooth_tchebycheff_eps))

    def _get_reference_point(
        self,
        sequence_reward_tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._tchebycheff_ref_point_fixed is not None:
            return torch.tensor(
                list(self._tchebycheff_ref_point_fixed),
                device=device,
                dtype=dtype,
            )

        if self._tchebycheff_ref_mode in ("batch_max", "batch"):
            return sequence_reward_tensor.amax(dim=0)

        # Default: running max over observed batches using sequence-level rewards.
        if not self._smooth_tchebycheff_allow_state_updates:
            if self._tchebycheff_ref_point is not None:
                return self._tchebycheff_ref_point.to(device=device, dtype=dtype)
            return sequence_reward_tensor.amax(dim=0)

        batch_max = sequence_reward_tensor.amax(dim=0).detach().cpu()
        if self._tchebycheff_ref_point is None:
            self._tchebycheff_ref_point = batch_max
        else:
            self._tchebycheff_ref_point = torch.maximum(self._tchebycheff_ref_point, batch_max)

        return self._tchebycheff_ref_point.to(device=device, dtype=dtype)

    def _aggregate_reward_tensor(self, reward_tensor: dict[str, torch.Tensor]):
        # (B, N, T)
        stacked_reward_tensor = torch.stack(list(reward_tensor.values()), dim=1)
        device = stacked_reward_tensor.device
        dtype = stacked_reward_tensor.dtype

        if self._smooth_tchebycheff_log_weights is None:
            self._smooth_tchebycheff_log_weights = self._init_log_weights(device=device, dtype=dtype).detach().cpu()

        preference_weights = self._preference_weights.to(device=device, dtype=dtype)
        log_weights = self._smooth_tchebycheff_log_weights.to(device=device, dtype=dtype)

        # Estimate R_i(theta) with batch mean sequence returns.
        sequence_reward_tensor = stacked_reward_tensor.sum(dim=-1)  # (B, N)
        objective_values = sequence_reward_tensor.mean(dim=0)  # (N,)
        reference_point = self._get_reference_point(sequence_reward_tensor, device=device, dtype=dtype)

        # Eq. (17): smooth Tchebycheff logits from weighted deficits.
        weighted_deficit = preference_weights * (reference_point - objective_values)
        smooth_weights = torch.softmax(log_weights / self._smooth_tchebycheff_temperature, dim=0)

        # Eq. (18): online mirror descent in log-weight space.
        # Use w_t to scalarize the current batch, then update to w_{t+1} for next batch.
        if self._smooth_tchebycheff_allow_state_updates:
            next_log_weights = log_weights + self._smooth_tchebycheff_omd_lr * weighted_deficit.detach()
            self._smooth_tchebycheff_log_weights = next_log_weights.detach().cpu()
        self._last_smooth_tchebycheff_weights = smooth_weights.detach().cpu()
        self._last_smooth_tchebycheff_objective_values = objective_values.detach().cpu()

        # PPO consumes token-level scores, so we mix token rewards with the smooth weights.
        scalarized_token_reward = torch.einsum("bnt,n->bt", stacked_reward_tensor, smooth_weights)
        return scalarized_token_reward, reference_point.detach().cpu().tolist()

    def _validate(self):
        prev_allow_state_updates = self._smooth_tchebycheff_allow_state_updates
        self._smooth_tchebycheff_allow_state_updates = False
        try:
            return super()._validate()
        finally:
            self._smooth_tchebycheff_allow_state_updates = prev_allow_state_updates

    def fit(self):
        """
        Same training loop as parent, with extra smooth-weight logging.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        self.extra_info = {}

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            base_tchebycheff.pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = base_tchebycheff.RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = base_tchebycheff.tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Training Progress"
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with base_tchebycheff.marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: base_tchebycheff.DataProto = base_tchebycheff.DataProto.from_single_dict(batch_dict)
                batch.meta_info["validate"] = False

                # add uid to batch
                batch.non_tensor_batch["uid"] = base_tchebycheff.np.array(
                    [str(base_tchebycheff.uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with base_tchebycheff.marked_timer("step", timing_raw):
                    # generate a batch
                    with base_tchebycheff.marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == base_tchebycheff.AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with base_tchebycheff.marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = base_tchebycheff.deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = base_tchebycheff.compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if "global_avg_length" in self.metrics.metrics:
                        response_length_batch = base_tchebycheff._compute_response_info(batch)["response_length"].tolist()
                        self.metrics.record_metric_many("global_avg_length", response_length_batch)
                        batch.non_tensor_batch["response_length_batch"] = base_tchebycheff.np.array(
                            response_length_batch, dtype=base_tchebycheff.np.int32
                        )

                    # we update the metrics after the batch is processed
                    batch.meta_info["extra_info"] = self.extra_info
                    log_extra_info = {f"{k}/{k}": v for k, v in self.extra_info.items()}
                    metrics.update(**log_extra_info)
                    self.extra_info.update(self.metrics.all_gather_metrics())

                    with base_tchebycheff.marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = base_tchebycheff.compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = base_tchebycheff.compute_reward(batch, self.reward_fn)

                    with base_tchebycheff.marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = base_tchebycheff.agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        with base_tchebycheff.marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with base_tchebycheff.marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = base_tchebycheff.ray.get(future_reward)

                    weighted_reward_tensor, ref_points = self._aggregate_reward_tensor(reward_tensor)
                    batch.batch["token_level_scores"] = weighted_reward_tensor

                    for i, z_i in enumerate(ref_points):
                        metrics.update({f"tchebycheff/ref_point_{i}": z_i})

                    # Extra logging for smooth Tchebycheff indicator weights.
                    if self._last_smooth_tchebycheff_weights is not None:
                        for i, w_i in enumerate(self._last_smooth_tchebycheff_weights.tolist()):
                            metrics.update({f"smooth_tchebycheff/weight_{i}": float(w_i)})
                    if self._last_smooth_tchebycheff_objective_values is not None:
                        for i, r_i in enumerate(self._last_smooth_tchebycheff_objective_values.tolist()):
                            metrics.update({f"smooth_tchebycheff/objective_{i}": float(r_i)})

                    with base_tchebycheff.marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: base_tchebycheff.np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = base_tchebycheff.apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = base_tchebycheff.compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.use_critic:
                        with base_tchebycheff.marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = base_tchebycheff.reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with base_tchebycheff.marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = base_tchebycheff.reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with base_tchebycheff.marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with base_tchebycheff.marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = base_tchebycheff.should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with base_tchebycheff.marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with base_tchebycheff.marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(base_tchebycheff.compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(base_tchebycheff.compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    base_tchebycheff.compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                if isinstance(self.train_dataloader.sampler, base_tchebycheff.AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    base_tchebycheff.pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
