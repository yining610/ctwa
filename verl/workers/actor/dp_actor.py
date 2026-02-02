# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from functools import partial
import math
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from verl.workers.actor.core_algos import compute_mgda_weights, compute_ctwa_stats

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def _compute_response_log_prob_current(self, data: DataProto) -> torch.Tensor:
        """
        Compute token-level log probs for the response tokens under the current actor policy.
        Used for post-update CTWA stats.
        
        Returns:
          torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        micro_batch_size = data.meta_info["micro_batch_size"]

        # Only keep what the forward needs
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask"]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=["uid"])

        if use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False
                )

            log_probs_lst.append(log_probs.to("cpu"))
            del micro_batch, model_inputs, log_probs

        log_probs = torch.concat(log_probs_lst, dim=0)
        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]

        if "per_objective_scores" in data.batch.keys():
            select_keys.append("per_objective_scores")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["uid"]
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    if loss_mode == "hack":
                        kwargs = {
                            "enable": True,
                            "per_objective_scores": model_inputs["per_objective_scores"],
                            "uid": model_inputs["uid"],
                            "objective_names": data.meta_info["objective_names"],
                        }
                    else:
                        kwargs = {}

                    out = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                        **kwargs,
                    )
                    if loss_mode == "hack":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_metrics = out
                        micro_batch_metrics.update(extra_metrics)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = out

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()

        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_covariance(self, data: DataProto):
        metrics = self.update_policy(data=data)

        assert "per_objective_scores" in data.batch.keys() and "uid" in data.non_tensor_batch, (
            "update_policy_covariance requires `data.batch['per_objective_scores']` and "
            "`data.non_tensor_batch['uid']`."
        )

        log_prob_new_full = self._compute_response_log_prob_current(data=data)

        # PPO clipping params
        clip_ratio = self.config.clip_ratio
        clip_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio

        w_agg = data.meta_info['ctwa_w_agg']
        global_metrics = data.meta_info['ctwa_global_metrics']

        ctwa_metrics = compute_ctwa_stats(
            old_log_prob=data.batch["old_log_probs"],
            log_prob_new=log_prob_new_full,
            advantages=data.batch["advantages"],
            response_mask=data.batch["response_mask"],
            per_objective_scores=data.batch["per_objective_scores"],
            uids=data.non_tensor_batch["uid"],
            cliprange_low=clip_low,
            cliprange_high=clip_high,
            w_agg=w_agg,
            global_metrics=global_metrics,
        )

        obj_names = data.meta_info["objective_names"]
        for j, name in enumerate(list(obj_names)):
            if f"ctwa/cov_mean/{j}" in ctwa_metrics:
                ctwa_metrics[f"ctwa/cov_mean/{name}"] = ctwa_metrics.pop(f"ctwa/cov_mean/{j}")
            if f"ctwa/corr_mean/{j}" in ctwa_metrics:
                ctwa_metrics[f"ctwa/corr_mean/{name}"] = ctwa_metrics.pop(f"ctwa/corr_mean/{j}")

        # Also log per-objective means for diagnostics
        u_mean = data.batch["per_objective_scores"].mean(dim=0).cpu().tolist()
        for j, name in enumerate(list(obj_names)):
            ctwa_metrics[f"ctwa/objective_mean/{name}"] = float(u_mean[j])

        append_to_dict(metrics, ctwa_metrics)

        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_mgda(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages_multi",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Prepare mini-batches as in standard PPO
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # count number of parameters on this replica
        num_params_per_layer = self.get_num_parameters()
        num_params = int(num_params_per_layer.sum().item())
        print(f"Number of parameters on replica {get_device_id()}: {num_params}")

        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # Build micro-batches for this mini-batch
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                advantages_multi_mb = mini_batch.batch["advantages_multi"]  # (B, T, S)
                S = advantages_multi_mb.shape[-1]
                grad_mat = torch.zeros((S, num_params), dtype=torch.float32, device=get_device_id())

                # Compute per-objective policy gradients
                for s in range(S):
                    self.actor_optimizer.zero_grad()

                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                        advantages_multi = model_inputs["advantages_multi"]  # (bsz, T, S)
                        advantages_s = advantages_multi[..., s]

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        calculate_entropy = entropy_coeff != 0
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        if on_policy:
                            old_log_prob_local = log_prob.detach()
                        else:
                            old_log_prob_local = old_log_prob

                        pg_loss_s, pg_clipfrac_s, ppo_kl_s, pg_clipfrac_lower_s = policy_loss_fn(
                            old_log_prob=old_log_prob_local,
                            log_prob=log_prob,
                            advantages=advantages_s,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_log_probs=rollout_log_probs,
                        )

                        # Entropy regularization
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss_s = pg_loss_s - entropy_loss * entropy_coeff
                        else:
                            policy_loss_s = pg_loss_s

                        # KL regularization
                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss_s = policy_loss_s + kl_loss * self.config.kl_loss_coef

                        loss_s = policy_loss_s * loss_scale_factor
                        loss_s.backward()
                        
                        micro_batch_metrics.update(
                            {
                                f"actor/pg_loss_{s}": pg_loss_s.detach().item() * loss_scale_factor,
                                f"actor/pg_clipfrac_{s}": pg_clipfrac_s.detach().item(),
                                f"actor/ppo_kl_{s}": ppo_kl_s.detach().item(),
                                f"actor/pg_clipfrac_lower_{s}": pg_clipfrac_lower_s.detach().item(),
                            }
                        )
                        append_to_dict(metrics, micro_batch_metrics)
                        
                    flat_chunks = []
                    for p in self.actor_module.parameters():
                        flat_chunks.append(p.grad.detach().to(torch.float32).view(-1))
                    grad_mat[s] = torch.cat(flat_chunks, dim=0)

                if self.config.norm_grad_norm:
                    row_norms = grad_mat.norm(dim=1, keepdim=True)
                    # For any objective with zero gradient norms (already converged), fall back to 1.0 so it stays unscaled.
                    row_norms = torch.where(row_norms > 0, row_norms, torch.ones_like(row_norms))
                    grad_mat = grad_mat / row_norms

                # Compute MGDA weights
                mgda_weights = compute_mgda_weights(grad_mat)  # (S,)

                mgda_metrics = {f"mgda/weight_{s}": float(mgda_weights[s].item()) for s in range(S)}
                append_to_dict(metrics, mgda_metrics)

                weighted_flat_grad = mgda_weights @ grad_mat

                # update parameters
                grad_metrics = self.apply_flat_gradients(
                    flat_grad_concat=weighted_flat_grad, num_params_per_layer=num_params_per_layer
                )
                append_to_dict(metrics, grad_metrics)

        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_lagrangian(self, data: DataProto):
        """PPO update using Lagrangian-weighted advantages."""

        advantages_multi = data.batch["advantages_multi"]  # (B, T, S)

        lambdas = torch.as_tensor(data.meta_info["lagrange_multipliers"]).to(advantages_multi.device)

        S = advantages_multi.shape[-1]
        primary_adv = advantages_multi[..., 0]  # (B, T)
        constraint_adv = advantages_multi[..., 1:]  # (B, T, K)
        combined_advantages = primary_adv + torch.einsum("btk,k->bt", constraint_adv, lambdas)

        # override advantages
        data.batch["advantages"] = combined_advantages
        return self.update_policy(data)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_gradnorm(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages_multi",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Prepare mini-batches as in standard PPO
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # Count number of parameters on this replica
        num_params_per_layer = self.get_num_parameters()
        num_params = int(num_params_per_layer.sum().item())
        
        entropy_coeff = self.config.entropy_coeff
        calculate_entropy = entropy_coeff != 0
        loss_agg_mode = self.config.loss_agg_mode
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        # Hyperparameters for GradNorm
        gradnorm_alpha = self.config.gradnorm_alpha
        gradnorm_lr = self.config.gradnorm_lr

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                # Build micro-batches for this mini-batch
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                advantages_multi_mb = mini_batch.batch["advantages_multi"]  # (B, T, S)
                S = advantages_multi_mb.shape[-1]

                device = get_device_id()
                grad_mat = torch.zeros((S, num_params), dtype=torch.float32, device=device)
                loss_vec = torch.zeros(S, dtype=torch.float32, device=device)

                # Compute per-objective gradients and losses
                for s in range(S):
                    self.actor_optimizer.zero_grad()

                    for micro_batch in micro_batches:
                        micro_batch = micro_batch.to(device)
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                        advantages_multi = model_inputs["advantages_multi"]  # (bsz, T, S)
                        advantages_s = advantages_multi[..., s]

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        if on_policy:
                            old_log_prob_local = log_prob.detach()
                        else:
                            old_log_prob_local = old_log_prob

                        pg_loss_s, pg_clipfrac_s, ppo_kl_s, pg_clipfrac_lower_s = policy_loss_fn(
                            old_log_prob=old_log_prob_local,
                            log_prob=log_prob,
                            advantages=advantages_s,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_log_probs=rollout_log_probs,
                        )

                        # Entropy regularization
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss_s = pg_loss_s - entropy_loss * entropy_coeff
                        else:
                            policy_loss_s = pg_loss_s

                        # KL regularization
                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss_s = policy_loss_s + kl_loss * self.config.kl_loss_coef

                        loss_s = policy_loss_s * loss_scale_factor
                        loss_s.backward()

                        # Accumulate scalar losses for GradNorm over the mini-batch
                        loss_vec[s] += policy_loss_s.detach() * loss_scale_factor

                        micro_batch_metrics.update(
                            {
                                f"gradnorm/pg_loss_{s}": pg_loss_s.detach().item() * loss_scale_factor,
                                f"gradnorm/pg_clipfrac_{s}": pg_clipfrac_s.detach().item(),
                                f"gradnorm/ppo_kl_{s}": ppo_kl_s.detach().item(),
                                f"gradnorm/pg_clipfrac_lower_{s}": pg_clipfrac_lower_s.detach().item(),
                            }
                        )
                        append_to_dict(metrics, micro_batch_metrics)

                    # Flatten gradients for this objective
                    flat_chunks = []
                    for p in self.actor_module.parameters():
                        flat_chunks.append(p.grad.detach().to(torch.float32).view(-1))
                    grad_mat[s] = torch.cat(flat_chunks, dim=0)

                grad_norms = grad_mat.norm(dim=1)  # (S,), L2 norm of the gradients

                # Initialize state on first use
                if not getattr(self, "_gradnorm_initialized", False):
                    self.gradnorm_L0 = loss_vec.detach().clone()
                    # TODO: try different initial weights
                    self.gradnorm_weights = torch.ones(S, dtype=torch.float32, device=device)
                    setattr(self, "_gradnorm_initialized", True)

                w = self.gradnorm_weights
                G = w * grad_norms
                G_avg = G.mean()

                loss_ratio = loss_vec / self.gradnorm_L0  # (S,), inverse training rate
                rel_loss = loss_ratio / loss_ratio.mean() # (S,), relative inverse training rate

                target_G = G_avg * (rel_loss ** gradnorm_alpha)

                # derivative of the L1 loss with respect to the weights
                w_grad = torch.sign(G - target_G) * grad_norms

                w = w - gradnorm_lr * w_grad
                w = w * (S / w.sum()) # normalization

                self.gradnorm_weights = w.detach()

                # Combine gradients using updated weights
                weighted_flat_grad = w @ grad_mat

                grad_metrics = self.apply_flat_gradients(
                    flat_grad_concat=weighted_flat_grad, num_params_per_layer=num_params_per_layer
                )
                append_to_dict(metrics, grad_metrics)

                gradnorm_metrics = {f"gradnorm/weight_{s}": float(w[s].item()) for s in range(S)}
                append_to_dict(metrics, gradnorm_metrics)

        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_pama(self, data: DataProto):
        """
        PAMA actor update:
        - The driver computes the closed-form scalar s* and passes it here.
        - Practically, we use s* as a non-negative scalar advantage applied to all valid response tokens.
        """
        self.actor_module.train()

        if "pama_s_star" not in data.meta_info:
            raise KeyError("PAMA requires `data.meta_info['pama_s_star']` (computed in ray_trainer_pama.py).")
        s_star = float(data.meta_info["pama_s_star"])
        temperature = data.meta_info["temperature"]

        # we follow original Noon PPO implementation
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "ref_log_prob",
        ]
        if self.config.tis_imp_ratio_cap > 0:
            raise NotImplementedError("PAMA update currently does not support TIS / rollout_log_probs.")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        entropy_coeff = self.config.entropy_coeff
        calculate_entropy = entropy_coeff != 0
        loss_agg_mode = self.config.loss_agg_mode

        # Noon PPO uses epsilon as the PPO clip ratio.
        clip_ratio = self.config.clip_ratio
        clip_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for _, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    response_mask = model_inputs["response_mask"]
                    ref_log_prob = model_inputs["ref_log_prob"]

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    neg_approx_kl = log_prob - ref_log_prob
                    neg_approx_kl = torch.clamp(neg_approx_kl, min=-20.0, max=20.0)
                    ratio = torch.exp(neg_approx_kl)

                    # Noon PPO objective: Equation (5)
                    ratio_clip = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
                    min_ratio = torch.minimum(ratio, ratio_clip)

                    # Scalar advantage, applied to each valid token
                    adv = s_star
                    pg_losses = -adv * min_ratio
                    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    if calculate_entropy:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss - entropy_coeff * entropy_loss
                    else:
                        policy_loss = pg_loss

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": float(pg_loss.detach().item()) * loss_scale_factor
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm.detach().item())})

        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_gradients(self, data: DataProto, target_layer_ids: list = None):

        self.actor_module.train()
        for p in self.actor_module.parameters():
            p.grad = None # clear previous gradients

        temperature = data.meta_info["temperature"]
        
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]

        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Single-pass, whole-batch gradient: split only for micro-batching and average across them
        micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
        total_batch_size = data.batch.batch_size[0]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = model_inputs["response_mask"]
            old_log_prob = model_inputs["old_log_probs"]
            rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
            advantages = model_inputs["advantages"]

            entropy_coeff = self.config.entropy_coeff
            loss_agg_mode = self.config.loss_agg_mode

            loss_scale_factor = response_mask.shape[0] / total_batch_size

            calculate_entropy = entropy_coeff != 0
            entropy, log_prob = self._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
            )

            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            policy_loss_fn = get_policy_loss_fn(loss_mode)

            if loss_mode == "hack":
                kwargs = {
                    "enable": False,
                    "per_objective_scores": None,
                    "uid": None,
                    "objective_names": None,
                }
            else:
                kwargs = {}
                
            out = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
                rollout_log_probs=rollout_log_probs,
                **kwargs,
            )

            if loss_mode == "hack":
                pg_loss, _, _, _, _ = out
            else:
                pg_loss, _, _, _ = out

            if entropy_coeff != 0:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = pg_loss - entropy_loss * entropy_coeff
            else:
                policy_loss = pg_loss

            if self.config.use_kl_loss:
                ref_log_prob = model_inputs["ref_log_prob"]
                kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

            loss = policy_loss * loss_scale_factor
            loss.backward()
        
        flat_grad_concat = []
        for p_name, p in self.actor_module.named_parameters():
            layer_id = int(p_name.split("layers.")[1].split(".")[0]) if "layers." in p_name else -1
            if target_layer_ids is not None and layer_id not in target_layer_ids:
                continue
            flat_grad_concat.append(p.grad.detach().to(torch.float32).view(-1))
        
        flat_grad_concat = torch.cat(flat_grad_concat)

        return flat_grad_concat

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def apply_flat_gradients(self, flat_grad_concat: torch.Tensor, num_params_per_layer: torch.Tensor):
        """Assign a flattened gradient vector to parameters and perform an optimizer step."""
        self.actor_module.train()
        self.actor_optimizer.zero_grad()

        offset = 0
        for idx, (_, p) in enumerate(self.actor_module.named_parameters()):
            expected = int(num_params_per_layer[idx].item())
            assert expected == p.numel(), f"Split size mismatch: expected {expected}, param has {p.numel()}"

            grad_slice = flat_grad_concat.narrow(dim=0, start=offset, length=expected)

            p.grad = grad_slice.to(device=p.device, dtype=p.dtype).view_as(p)
            offset += expected

        grad_norm = self._optimizer_step()
        self.actor_optimizer.zero_grad()
        return {"actor/grad_norm": float(grad_norm.detach().item())}

    def get_num_parameters(self, target_layer_ids: list = None):
        num_params_per_layer = []
        for p_name, p in self.actor_module.named_parameters():
            layer_id = int(p_name.split("layers.")[1].split(".")[0]) if "layers." in p_name else -1
            if target_layer_ids is not None and layer_id not in target_layer_ids:
                continue
            num_params_per_layer.append(p.numel())
        return torch.tensor(num_params_per_layer, dtype=torch.int32)