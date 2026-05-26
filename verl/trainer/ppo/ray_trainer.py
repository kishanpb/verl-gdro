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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, List, Dict, Tuple

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.group_dro_grpo import ClassDroExp3p, Exp3pConfig, PassKOnlineClassifier
from verl.utils.budget_allocation import (
    budget_allocation_vanilla,
    budget_allocation_knapsack,
    budget_allocation_knapsack_group_dro,
    budget_allocation_knapsack_gdro_passk,
)

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using standard Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        # Prompt-GDRO (Problem 1) state
        algo = self.config.algorithm
        self.gdro_enabled = bool(algo.get("gdro_enable", False))
        # If False, we still compute bins / update EXP3 state, but we do NOT apply GDRO weights to the actor loss.
        # This is useful for "prob2.1-only" experiments that want pass@k bins and rollout allocation without
        # reweighting the GRPO update.
        self.gdro_apply_weights = bool(algo.get("gdro_apply_weights", True))
        if self.gdro_enabled:
            cfg = Exp3pConfig(
                eta_q=float(algo.get("gdro_eta_q", 0.1)),
                gamma=float(algo.get("gdro_gamma", 0.15)),
                max_class_weight=float(algo.get("gdro_max_class_weight", 10.0)),
                prompt_classifier=str(algo.get("gdro_prompt_classifier", "gsm8k")),
                debias_scores=bool(algo.get("gdro_debias_scores", False)),
                debias_scores_ema=bool(algo.get("gdro_debias_scores_ema", False)),
                ema_beta=float(algo.get("gdro_ema_beta", 0.1)),
                use_zscore=bool(algo.get("gdro_use_zscore", False)),
                z_std_floor=float(algo.get("gdro_z_std_floor", 1e-3)),
                z_cap=float(algo.get("gdro_z_cap", 3.0)),
                passk_num_bins=int(algo.get("gdro_passk_num_bins", algo.get("passk_num_bins", 10))),
                passk_history_len=int(algo.get("gdro_passk_history_len", algo.get("passk_history_len", 0))),
                passk_hysteresis=float(algo.get("gdro_passk_hysteresis", algo.get("passk_hysteresis", 0.0))),
                passk_edges=str(algo.get("gdro_passk_edges", algo.get("passk_edges", ""))),
                passk_exclude_extremes=bool(
                    algo.get("gdro_passk_exclude_extremes", algo.get("passk_exclude_extremes", False))
                ),
                loss_norm_by_class=bool(algo.get("gdro_loss_norm_by_class", algo.get("loss_norm_by_class", False))),
                passk_focus_enable=bool(algo.get("gdro_passk_focus_enable", algo.get("passk_focus_enable", False))),
                passk_focus_map=str(algo.get("gdro_passk_focus_map", algo.get("passk_focus_map", ""))),
                passk_focus_warmup_steps=int(
                    algo.get("gdro_passk_focus_warmup_steps", algo.get("passk_focus_warmup_steps", 0))
                ),
                passk_focus_ramp_steps=int(
                    algo.get("gdro_passk_focus_ramp_steps", algo.get("passk_focus_ramp_steps", 0))
                ),
            )
            self.gdro = ClassDroExp3p(cfg)
            self.gdro_weight_mode = str(algo.get("gdro_weight_mode", "class")).lower()
            # Optional linear schedules for eta_q and gamma
            try:
                self._gdro_sched = {
                    "eta_q": {
                        "start": float(algo.get("gdro_eta_q_start", cfg.eta_q)),
                        "end": float(algo.get("gdro_eta_q_final", cfg.eta_q)),
                        "t0": int(algo.get("gdro_eta_q_step_start", 0)),
                        "t1": int(algo.get("gdro_eta_q_step_end", 0)),
                    },
                    "gamma": {
                        "start": float(algo.get("gdro_gamma_start", cfg.gamma)),
                        "end": float(algo.get("gdro_gamma_final", cfg.gamma)),
                        "t0": int(algo.get("gdro_gamma_step_start", 0)),
                        "t1": int(algo.get("gdro_gamma_step_end", 0)),
                    },
                }
            except Exception:
                self._gdro_sched = {"eta_q": {"t1": 0}, "gamma": {"t1": 0}}
        else:
            self.gdro = None
            self.gdro_weight_mode = "class"

        # Rollout-GDRO (Problem 2.1) state - minimal implementation
        self.rollout_budget_mode = str(algo.get("rollout_budget_mode", "uniform")).lower()
        self.rollout_budget_n_min = max(1, int(algo.get("rollout_budget_n_min", 2)))
        rollout_base_n = max(1, int(self.config.actor_rollout_ref.rollout.n))
        max_multiplier = float(algo.get("rollout_budget_n_max_multiplier", 2.0))
        default_n_max = int(round(rollout_base_n * max_multiplier))
        self.rollout_budget_n_max = int(algo.get("rollout_budget_n_max", default_n_max))
        if self.rollout_budget_n_max < self.rollout_budget_n_min:
            self.rollout_budget_n_max = self.rollout_budget_n_min
        # Rollout-GDRO (Problem 2.1) controls
        self.rollout_prob21_enable = bool(algo.get("rollout_budget_prob21", algo.get("rollout_prob21_enable", False)))
        self.rollout_budget_dual_mu = float(algo.get("rollout_budget_dual_mu", 0.0))
        self.rollout_budget_dual_lr = float(algo.get("rollout_budget_dual_lr", 0.05))
        # Rollout-GDRO can work independently of Prompt-GDRO
        self.rollout_allocator = None
        self._rollout_step_state: Dict[str, Optional[List[str]]] = {}
        self._rollout_allocator_shared_passk = False
        self._rollout_allocator_fallbacks: int = 0
        # Rollout-GDRO v3: Store previous step's classification for AFTER-generation approach
        # At step N, we use step N-1's classification to allocate rollouts
        # After generation at step N, we classify and store for step N+1
        self._rollout_prev_step_class_ids: Optional[List[str]] = None
        self._rollout_prev_step_weights: Optional[torch.Tensor] = None
        self._rollout_prev_step_weight_map: Optional[Dict[str, float]] = None
        if self.rollout_budget_mode == "groupdro":
            if self.gdro is not None:
                # Share the same EXP3.P instance (prompt sampling + rollout allocation)
                self.rollout_allocator = self.gdro
                self._rollout_allocator_shared_passk = True
            else:
                rollout_cfg = Exp3pConfig(
                    eta_q=float(algo.get("rollout_budget_eta", algo.get("gdro_eta_q", 0.65))),
                    gamma=float(algo.get("rollout_budget_gamma", algo.get("gdro_gamma", 0.01))),
                    max_class_weight=float(
                        algo.get("rollout_budget_max_class_weight", algo.get("gdro_max_class_weight", 15.0))
                    ),
                    prompt_classifier=str(
                        algo.get("rollout_budget_classifier", algo.get("gdro_prompt_classifier", "passk_online"))
                    ),
                    debias_scores=bool(algo.get("rollout_budget_debias_scores", algo.get("gdro_debias_scores", False))),
                    debias_scores_ema=bool(
                        algo.get("rollout_budget_debias_scores_ema", algo.get("gdro_debias_scores_ema", True))
                    ),
                    ema_beta=float(algo.get("rollout_budget_ema_beta", algo.get("gdro_ema_beta", 0.12))),
                    passk_num_bins=int(
                        algo.get(
                            "rollout_budget_passk_num_bins",
                            algo.get("gdro_passk_num_bins", algo.get("passk_num_bins", 10)),
                        )
                    ),
                    passk_history_len=int(
                        algo.get(
                            "rollout_budget_passk_history_len",
                            algo.get("gdro_passk_history_len", algo.get("passk_history_len", 0)),
                        )
                    ),
                    passk_hysteresis=float(
                        algo.get(
                            "rollout_budget_passk_hysteresis",
                            algo.get("gdro_passk_hysteresis", algo.get("passk_hysteresis", 0.0)),
                        )
                    ),
                    passk_edges=str(
                        algo.get("rollout_budget_passk_edges", algo.get("gdro_passk_edges", algo.get("passk_edges", "")))
                    ),
                    passk_exclude_extremes=bool(
                        algo.get(
                            "rollout_budget_passk_exclude_extremes",
                            algo.get("gdro_passk_exclude_extremes", algo.get("passk_exclude_extremes", False)),
                        )
                    ),
                    loss_norm_by_class=bool(
                        algo.get(
                            "rollout_budget_loss_norm_by_class",
                            algo.get("gdro_loss_norm_by_class", algo.get("loss_norm_by_class", False)),
                        )
                    ),
                    passk_focus_enable=bool(
                        algo.get(
                            "rollout_budget_passk_focus_enable",
                            algo.get("gdro_passk_focus_enable", algo.get("passk_focus_enable", False)),
                        )
                    ),
                    passk_focus_map=str(
                        algo.get("rollout_budget_passk_focus_map", algo.get("gdro_passk_focus_map", algo.get("passk_focus_map", "")))
                    ),
                    passk_focus_warmup_steps=int(
                        algo.get(
                            "rollout_budget_passk_focus_warmup_steps",
                            algo.get("gdro_passk_focus_warmup_steps", algo.get("passk_focus_warmup_steps", 0)),
                        )
                    ),
                    passk_focus_ramp_steps=int(
                        algo.get(
                            "rollout_budget_passk_focus_ramp_steps",
                            algo.get("gdro_passk_focus_ramp_steps", algo.get("passk_focus_ramp_steps", 0)),
                        )
                    ),
                )
                self.rollout_allocator = ClassDroExp3p(rollout_cfg)
        if self.rollout_allocator is not None and self.rollout_prob21_enable:
            # Configure discrete rollout arms for Problem 2.1
            try:
                arms = list(range(self.rollout_budget_n_min, self.rollout_budget_n_max + 1))
                if hasattr(self.rollout_allocator, "set_rollout_arms"):
                    self.rollout_allocator.set_rollout_arms(arms)
            except Exception:
                pass

        # Problem 3 (off-policy GRPO update distribution)
        self.offpolicy_grpo_enable = bool(algo.get("offpolicy_grpo_enable", False))
        self.offpolicy_w_min = max(1e-4, float(algo.get("offpolicy_grpo_w_min", 0.5)))
        self.offpolicy_w_max = max(self.offpolicy_w_min, float(algo.get("offpolicy_grpo_w_max", 2.0)))
        self.offpolicy_eta = max(0.0, float(algo.get("offpolicy_grpo_eta", 0.2)))
        self.offpolicy_classifier: Optional[ClassDroExp3p] = None
        self._offpolicy_classifier_private: Optional[ClassDroExp3p] = None
        self.offpolicy_bin_weights: Dict[str, float] = {}
        self._prob3_expanded_class_ids: Optional[List[str]] = None
        self._prob3_prompt_class_counts: Optional[Dict[str, int]] = None
        self._prob3_prompt_mass: Optional[Dict[str, float]] = None
        self._prob3_sample_weights: Optional[torch.Tensor] = None
        if self.offpolicy_grpo_enable:
            self.offpolicy_classifier = self._init_offpolicy_classifier(algo)

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if config.actor_rollout_ref.actor.strategy == "megatron":
                model_parallel_size = (
                    config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                    * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
                )
                assert (
                    n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
                ), (
                    f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                    f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
                )
                megatron_dp = n_gpus // (
                    model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
                )
                minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            else:
                minimal_bsz = n_gpus

            # 1. Check total batch size for data correctness
            real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
            assert real_train_batch_size % minimal_bsz == 0, (
                f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
                f"({minimal_bsz})"
            )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        # numpy already imported at top as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        else:
            # Ensure generation batch keeps reward-context fields needed by budget allocation (e.g., pass@k lookup)
            for key in ("data_source", "extra_info", "uid", "reward_model"):
                if key in batch.non_tensor_batch and key not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch[key] = batch.non_tensor_batch[key]

        return gen_batch

    def _prepare_classification_inputs(self, batch: DataProto) -> Tuple[List[str], List[Optional[dict]]]:
        """Prepare prompt texts and metadata for classification (unified for Prompt-GDRO and Rollout-GDRO).
        
        Args:
            batch: DataProto batch (can be before or after generation)
            
        Returns:
            prompt_texts: List of decoded prompt strings
            metadatas: List of metadata dicts (with uid, extra_info fields if available)
        """
        batch_size = len(batch)
        prompt_texts: List[str] = []
        
        # Extract prompt texts
        prompts_ids = batch.batch.get("prompts", None)
        if prompts_ids is not None:
            try:
                prompt_texts = self.tokenizer.batch_decode(prompts_ids, skip_special_tokens=True)
            except Exception:
                prompt_texts = []
        else:
            # Fallback: decode the prompt portion from input_ids (exclude response tokens if present)
            try:
                input_ids = batch.batch.get("input_ids", None)
                attention_mask = batch.batch.get("attention_mask", None)
                responses = batch.batch.get("responses", None)
                if input_ids is not None and attention_mask is not None:
                    if responses is not None:
                        resp_len = responses.shape[1]
                        prompt_ids = input_ids[:, :-resp_len]
                        prompt_mask = attention_mask[:, :-resp_len]
                    else:
                        prompt_ids = input_ids
                        prompt_mask = attention_mask
                    prompt_texts = []
                    for i in range(prompt_ids.shape[0]):
                        ids_i = prompt_ids[i][prompt_mask[i].bool()].tolist()
                        text = self.tokenizer.decode(ids_i, skip_special_tokens=True)
                        prompt_texts.append(text)
                else:
                    prompt_texts = [""] * batch_size
            except Exception:
                prompt_texts = [""] * batch_size
        
        if not prompt_texts or len(prompt_texts) != batch_size:
            prompt_texts = [""] * batch_size
        
        # Prepare metadata list
        metadatas_source = batch.non_tensor_batch.get("metadata", None)
        if metadatas_source is None:
            metadatas = [None] * batch_size
        else:
            try:
                metadatas = metadatas_source.tolist() if hasattr(metadatas_source, "tolist") else list(metadatas_source)
            except Exception:
                metadatas = list(metadatas_source)

        if len(metadatas) < batch_size:
            metadatas.extend([None] * (batch_size - len(metadatas)))
        elif len(metadatas) > batch_size:
            metadatas = metadatas[:batch_size]
        
        # Enrich metadata with extra_info fields if present
        try:
            if "extra_info" in batch.non_tensor_batch:
                extra_infos = batch.non_tensor_batch["extra_info"]
                extra_list = extra_infos.tolist() if hasattr(extra_infos, "tolist") else list(extra_infos)
                for i, ei in enumerate(extra_list):
                    if i < len(metadatas):
                        if not isinstance(metadatas[i], dict):
                            metadatas[i] = {} if metadatas[i] is None else dict(metadatas[i])
                        if isinstance(ei, dict):
                            lvl = ei.get("level", None)
                            typ = ei.get("type", None)
                            if lvl is not None:
                                metadatas[i]["level"] = lvl
                            if typ is not None:
                                metadatas[i]["type"] = typ
        except Exception:
            pass
        
        # Inject lenbin if available
        try:
            if "lenbin" in batch.non_tensor_batch:
                lb = batch.non_tensor_batch["lenbin"]
                lb_list = lb.tolist() if hasattr(lb, "tolist") else list(lb)
                for i, val in enumerate(lb_list):
                    if i < len(metadatas):
                        if not isinstance(metadatas[i], dict):
                            metadatas[i] = {} if metadatas[i] is None else dict(metadatas[i])
                        metadatas[i]["lenbin"] = str(val)
        except Exception:
            pass
        
        # Inject UIDs (critical for passk_online classifier)
        uids_arr = batch.non_tensor_batch.get("uid", None)
        if uids_arr is not None:
            try:
                uids_list = uids_arr.tolist() if hasattr(uids_arr, "tolist") else list(uids_arr)
                for i, u in enumerate(uids_list):
                    if i < len(metadatas):
                        if not isinstance(metadatas[i], dict):
                            metadatas[i] = {} if metadatas[i] is None else dict(metadatas[i])
                        if u is not None:
                            metadatas[i]["uid"] = str(u)
            except Exception:
                pass
        
        return prompt_texts, metadatas

    def _init_offpolicy_classifier(self, algo) -> Optional[ClassDroExp3p]:
        """Instantiate the classifier used by Problem 3 when no shared state exists."""
        if self.gdro is not None:
            return self.gdro
        if self.rollout_allocator is not None:
            return self.rollout_allocator
        cfg = Exp3pConfig(
            eta_q=float(algo.get("offpolicy_grpo_eta_q", algo.get("gdro_eta_q", 0.65))),
            gamma=float(algo.get("offpolicy_grpo_gamma", algo.get("gdro_gamma", 0.01))),
            max_class_weight=float(algo.get("offpolicy_grpo_max_class_weight", algo.get("gdro_max_class_weight", 15.0))),
            prompt_classifier=str(algo.get("offpolicy_grpo_classifier", algo.get("gdro_prompt_classifier", "passk_online"))),
            debias_scores=bool(algo.get("offpolicy_grpo_debias_scores", algo.get("gdro_debias_scores", False))),
            debias_scores_ema=bool(algo.get("offpolicy_grpo_debias_scores_ema", algo.get("gdro_debias_scores_ema", False))),
            ema_beta=float(algo.get("offpolicy_grpo_ema_beta", algo.get("gdro_ema_beta", 0.1))),
            passk_num_bins=int(algo.get("offpolicy_grpo_passk_num_bins", algo.get("gdro_passk_num_bins", algo.get("passk_num_bins", 10)))),
            passk_history_len=int(algo.get("offpolicy_grpo_passk_history_len", algo.get("gdro_passk_history_len", algo.get("passk_history_len", 0)))),
            passk_hysteresis=float(algo.get("offpolicy_grpo_passk_hysteresis", algo.get("gdro_passk_hysteresis", algo.get("passk_hysteresis", 0.0)))),
            passk_edges=str(algo.get("offpolicy_grpo_passk_edges", algo.get("gdro_passk_edges", algo.get("passk_edges", "")))),
            passk_exclude_extremes=bool(algo.get("offpolicy_grpo_passk_exclude_extremes", algo.get("gdro_passk_exclude_extremes", algo.get("passk_exclude_extremes", False)))),
            loss_norm_by_class=bool(algo.get("offpolicy_grpo_loss_norm_by_class", algo.get("gdro_loss_norm_by_class", algo.get("loss_norm_by_class", False)))),
            passk_focus_enable=bool(algo.get("offpolicy_grpo_passk_focus_enable", algo.get("gdro_passk_focus_enable", algo.get("passk_focus_enable", False)))),
            passk_focus_map=str(algo.get("offpolicy_grpo_passk_focus_map", algo.get("gdro_passk_focus_map", algo.get("passk_focus_map", "")))),
            passk_focus_warmup_steps=int(algo.get("offpolicy_grpo_passk_focus_warmup_steps", algo.get("gdro_passk_focus_warmup_steps", algo.get("passk_focus_warmup_steps", 0)))),
            passk_focus_ramp_steps=int(algo.get("offpolicy_grpo_passk_focus_ramp_steps", algo.get("gdro_passk_focus_ramp_steps", algo.get("passk_focus_ramp_steps", 0)))),
        )
        classifier = ClassDroExp3p(cfg)
        self._offpolicy_classifier_private = classifier
        return classifier

    def _reset_prob3_state(self):
        self._prob3_expanded_class_ids = None
        self._prob3_prompt_class_counts = None
        self._prob3_prompt_mass = None
        self._prob3_sample_weights = None
        self._prob3_behavior_log_probs_active = False

    def _extract_prompt_bins(
        self, expanded_class_ids: List[str], n_rollouts: int
    ) -> Tuple[List[str], Dict[str, int], Dict[str, float]]:
        if not expanded_class_ids:
            return [], {}, {}
        if n_rollouts <= 0:
            n_rollouts = 1
        if len(expanded_class_ids) % n_rollouts != 0:
            per_prompt = list(expanded_class_ids)
        else:
            num_prompts = len(expanded_class_ids) // n_rollouts
            per_prompt = [expanded_class_ids[i * n_rollouts] for i in range(num_prompts)]
        counts = Counter(per_prompt)
        total = float(sum(counts.values())) or 1.0
        mass = {cid: float(cnt) / total for cid, cnt in counts.items()}
        return per_prompt, counts, mass

    def _build_prob3_weight_vector(self, class_ids: List[str]) -> Optional[torch.Tensor]:
        if not class_ids:
            return None
        weights = []
        for cid in class_ids:
            w = float(self.offpolicy_bin_weights.get(cid, 1.0))
            weights.append(w)
        if not weights:
            return None
        return torch.tensor(weights, dtype=torch.float32)

    def _update_prob3_state(self, class_ids: Optional[List[str]], metrics: dict):
        if not class_ids:
            self._reset_prob3_state()
            return
        self._prob3_expanded_class_ids = list(class_ids)
        n_rollouts = int(self.config.actor_rollout_ref.rollout.n)
        _, counts, mass = self._extract_prompt_bins(class_ids, n_rollouts)
        self._prob3_prompt_class_counts = counts
        self._prob3_prompt_mass = mass
        for cid in mass:
            self.offpolicy_bin_weights.setdefault(cid, 1.0)
        weight_vec = self._build_prob3_weight_vector(class_ids)
        self._prob3_sample_weights = weight_vec
        if mass:
            metrics["offpolicy/num_classes"] = len(mass)

    def _apply_prob3_behavior_log_probs(self, batch: DataProto, metrics: dict):
        if not self.offpolicy_grpo_enable:
            batch.batch.pop("behavior_log_probs", None)
            self._prob3_behavior_log_probs_active = False
            return
        try:
            rollout_log_probs = batch.batch["rollout_log_probs"]
        except Exception:
            rollout_log_probs = None
        weights = self._prob3_sample_weights
        if rollout_log_probs is None or weights is None:
            batch.batch.pop("behavior_log_probs", None)
            self._prob3_behavior_log_probs_active = False
            metrics["offpolicy/behavior_log_probs_applied"] = 0.0
            metrics["offpolicy/rollout_log_probs_present"] = 0.0 if rollout_log_probs is None else 1.0
            return
        beh_weights = weights.to(device=rollout_log_probs.device, dtype=rollout_log_probs.dtype)
        stable_w = torch.clamp(beh_weights, min=self.offpolicy_w_min)
        log_w = torch.log(stable_w).unsqueeze(-1)
        behavior_log_probs = rollout_log_probs - log_w
        batch.batch["behavior_log_probs"] = behavior_log_probs
        self._prob3_behavior_log_probs_active = True
        metrics["offpolicy/behavior_log_probs_applied"] = 1.0
        metrics["offpolicy/rollout_log_probs_present"] = 1.0
        metrics["offpolicy/bin_weight_mean"] = float(beh_weights.mean().item())
        metrics["offpolicy/bin_weight_min"] = float(beh_weights.min().item())
        metrics["offpolicy/bin_weight_max"] = float(beh_weights.max().item())

    def _project_offpolicy_weights(self, mass: Dict[str, float]):
        if not mass:
            return
        base_weights = {}
        for cid in mass:
            base_weights[cid] = float(self.offpolicy_bin_weights.get(cid, 1.0))
        denom = sum(mass[cid] * base_weights[cid] for cid in mass)
        if denom <= 0:
            denom = 1.0
        scale = 1.0 / denom
        scaled = {}
        for cid, base in base_weights.items():
            val = base * scale
            val = max(self.offpolicy_w_min, min(self.offpolicy_w_max, val))
            scaled[cid] = val
        denom = sum(mass[cid] * scaled[cid] for cid in mass)
        if denom > 0:
            scale = 1.0 / denom
            for cid, val in scaled.items():
                adjusted = val * scale
                adjusted = max(self.offpolicy_w_min, min(self.offpolicy_w_max, adjusted))
                self.offpolicy_bin_weights[cid] = adjusted
        else:
            for cid, val in scaled.items():
                self.offpolicy_bin_weights[cid] = val

    def _update_offpolicy_weights(self, per_sample_lb: torch.Tensor, metrics: dict, batch: Optional[DataProto] = None):
        if not self.offpolicy_grpo_enable:
            return
        if not bool(getattr(self, "_prob3_behavior_log_probs_active", False)):
            metrics["offpolicy/skip_w_update_no_behavior_log_probs"] = 1.0
            return
        expanded_ids = self._prob3_expanded_class_ids
        mass = self._prob3_prompt_mass
        if per_sample_lb is None or not expanded_ids or not mass:
            return
        try:
            losses = per_sample_lb.detach().cpu().numpy().tolist()
        except Exception:
            return
        by_class: Dict[str, List[float]] = defaultdict(list)
        for cid, lb in zip(expanded_ids, losses):
            by_class[cid].append(float(lb))
        for cid, vals in by_class.items():
            if not vals:
                continue
            mean_lb = float(np.mean(vals))
            self.offpolicy_bin_weights.setdefault(cid, 1.0)
            self.offpolicy_bin_weights[cid] = self.offpolicy_bin_weights[cid] * math.exp(self.offpolicy_eta * mean_lb)
        self._project_offpolicy_weights(mass)
        active_weights = {cid: float(self.offpolicy_bin_weights.get(cid, 1.0)) for cid in mass}
        if active_weights:
            w_tensor = torch.tensor(list(active_weights.values()), dtype=torch.float32)
            metrics["offpolicy/active_weight_mean"] = float(w_tensor.mean().item())
            metrics["offpolicy/active_weight_min"] = float(w_tensor.min().item())
            metrics["offpolicy/active_weight_max"] = float(w_tensor.max().item())
            metrics["offpolicy/denom_calib_spread"] = float((w_tensor.max() - w_tensor.min()).item())
            w_sum = float(w_tensor.sum().item())
            if w_sum > 0.0:
                p = w_tensor / w_sum
                entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
                metrics["offpolicy/denom_calib_entropy"] = float(entropy)
            prev_weights = getattr(self, "_offpolicy_prev_weights", None)
            if isinstance(prev_weights, dict):
                l2_delta = math.sqrt(
                    sum((active_weights[cid] - float(prev_weights.get(cid, active_weights[cid]))) ** 2 for cid in active_weights)
                )
            else:
                l2_delta = 0.0
            metrics["offpolicy/denom_calib_l2_delta"] = float(l2_delta)
            self._offpolicy_prev_weights = dict(active_weights)
            for cid, w in active_weights.items():
                metrics[f"offpolicy/denom_calib_w@bin/{cid}"] = float(w)
                metrics[f"offpolicy/effective_mass@bin/{cid}"] = float(mass.get(cid, 0.0) * w)
            if batch is not None:
                try:
                    old_log_probs = batch.batch["old_log_probs"]
                    rollout_log_probs = batch.batch["rollout_log_probs"]
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    response_mask = batch.batch["response_mask"]
                    if (
                        old_log_probs is not None
                        and rollout_log_probs is not None
                        and response_mask is not None
                        and old_log_probs.shape == rollout_log_probs.shape
                        and response_mask.shape == old_log_probs.shape
                    ):
                        delta_logp = (old_log_probs - rollout_log_probs) * response_mask
                        denom = response_mask.sum(dim=-1).clamp(min=1)
                        per_sample_log_ratio = (delta_logp.sum(dim=-1) / denom).detach().cpu().numpy().tolist()
                        if len(per_sample_log_ratio) == len(expanded_ids):
                            by_class_lr: Dict[str, List[float]] = defaultdict(list)
                            for cid, lr in zip(expanded_ids, per_sample_log_ratio):
                                by_class_lr[cid].append(float(lr))
                            log_ratio_means: Dict[str, float] = {}
                            for cid, vals in by_class_lr.items():
                                if vals:
                                    log_ratio_means[cid] = float(np.mean(vals))
                                    metrics[f"offpolicy/log_ratio_mean@bin/{cid}"] = float(log_ratio_means[cid])
                            pairs = [
                                (active_weights[cid], log_ratio_means[cid])
                                for cid in active_weights
                                if cid in log_ratio_means and np.isfinite(log_ratio_means[cid])
                            ]
                            if len(pairs) >= 2:
                                w_vals, lr_vals = zip(*pairs, strict=False)
                                corr = float(np.corrcoef(np.array(w_vals), np.array(lr_vals))[0, 1])
                                if np.isfinite(corr):
                                    metrics["offpolicy/corr_denom_calib_vs_log_ratio"] = corr
                except Exception:
                    pass

    def _allocate_rollouts_by_group(
        self,
        class_ids: List[str],
        weights: torch.Tensor,
        total_budget: int,
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
        """Allocate rollout counts per prompt based on exp3p weights (Rollout-GDRO - minimal implementation).
        
        Args:
            class_ids: List of class/bin labels for each prompt
            weights: exp3p weights per prompt (from weights_for_samples)
            total_budget: Total number of rollouts = batch_size * rollout.n
            
        Returns:
            budgets: Rollout count per original prompt (n_b)
            selected_indices: Indices into original batch (with repeats for expansion)
            expanded_class_ids: Class IDs for expanded batch (for update mapping)
        """
        num_samples = len(class_ids)
        if num_samples == 0 or total_budget <= 0:
            return None, None, None

        n_min = int(self.rollout_budget_n_min)
        n_max = int(self.rollout_budget_n_max)
        base_budget = n_min * num_samples
        if total_budget < base_budget:
            return None, None, None

        # Start with n_min per prompt
        budgets = np.full(num_samples, n_min, dtype=int)
        remaining = total_budget - base_budget
        capacity_per_prompt = max(0, n_max - n_min)
        capacity = np.full(num_samples, capacity_per_prompt, dtype=int)

        # Normalize weights
        weights_np = weights.detach().cpu().numpy().astype(float)
        if not np.all(np.isfinite(weights_np)) or np.sum(weights_np) <= 0:
            weights_np = np.ones(num_samples, dtype=float)
        weights_np = np.maximum(weights_np, 1e-8)
        weights_np = weights_np / weights_np.sum()

        # Distribute remaining budget proportionally to weights
        if remaining > 0 and capacity_per_prompt > 0:
            raw_extra = weights_np * remaining
            clipped_raw = np.minimum(raw_extra, capacity.astype(float))
            extras = np.floor(clipped_raw).astype(int)
            leftover = int(remaining - extras.sum())
            
            # Distribute leftover using fractional parts
            if leftover > 0:
                fractional = clipped_raw - np.floor(clipped_raw)
                order = np.argsort(-fractional)
                for idx in order:
                    if leftover == 0:
                        break
                    if extras[idx] >= capacity[idx]:
                        continue
                    extras[idx] += 1
                    leftover -= 1
            
            # Final leftover distribution
            if leftover > 0:
                for idx in range(num_samples):
                    if leftover == 0:
                        break
                    available = capacity[idx] - extras[idx]
                    if available <= 0:
                        continue
                    take = min(available, leftover)
                    extras[idx] += take
                    leftover -= take
            
            budgets = budgets + extras

        # Validate bounds
        if budgets.min() < n_min or budgets.max() > n_max:
            return None, None, None
        
        # Ensure exact budget match
        diff = int(total_budget - budgets.sum())
        if diff != 0:
            adjust_order = np.argsort(weights_np * (-1 if diff > 0 else 1))
            for idx in adjust_order:
                if diff == 0:
                    break
                if diff > 0:
                    available = capacity[idx] - (budgets[idx] - n_min)
                    if available <= 0:
                        continue
                    take = min(available, diff)
                    budgets[idx] += take
                    diff -= take
                else:
                    reducible = budgets[idx] - n_min
                    if reducible <= 0:
                        continue
                    take = min(reducible, -diff)
                    budgets[idx] -= take
                    diff += take
        
        if budgets.sum() != total_budget:
            return None, None, None

        # Expand batch: create selected_indices with repeats
        selected_indices: List[int] = []
        for idx, count in enumerate(budgets.tolist()):
            if count <= 0:
                continue
            selected_indices.extend([idx] * int(count))
        
        if len(selected_indices) != total_budget:
            return None, None, None

        expanded_class_ids = [class_ids[idx] for idx in selected_indices]
        return budgets, selected_indices, expanded_class_ids

    def _validate(self):
        # Reset validation print counter for this validation run
        if hasattr(self.val_reward_fn, '_printed_in_validation_count'):
            self.val_reward_fn._printed_in_validation_count = 0
        
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_extra_infos = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            # collect per-sample extra_info for grouping (e.g., MATH level/type)
            try:
                if "extra_info" in test_batch.non_tensor_batch:
                    eis = test_batch.non_tensor_batch["extra_info"]
                    # Convert to list of dicts if needed
                    if hasattr(eis, "tolist"):
                        eis = eis.tolist()
                    # Ensure length matches number of samples in this batch
                    if isinstance(eis, list):
                        # Some datasets store one extra_info per original prompt; repeat if necessary
                        if len(eis) and len(eis) != reward_tensor.shape[0]:
                            # Repeat each entry to match interleaved validation repeats
                            repeats = self.config.actor_rollout_ref.rollout.val_kwargs.n
                            expanded = []
                            for ei in eis:
                                expanded.extend([ei] * repeats)
                            eis = expanded
                        sample_extra_infos.extend(eis)
            except Exception:
                pass

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    # Keep only core summary metrics; drop all val-aux metrics
                    if (
                        (var_name == core_var)
                        and (metric_name.startswith("mean") or metric_name.startswith("best"))
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_dict[f"val-core/{data_source}/{var_name}/{metric_name}"] = metric_val

        # NOTE: Drop dataset-level aggregation (expensive). We'll emit only compact metrics below.

        # Drop auxiliary num_turns logging to reduce overhead

        # Optional: evaluation-only subgroup metrics (e.g., MATH type×level)
        try:
            if bool(self.config.algorithm.get("eval_group_metrics_enable", False)) and len(sample_inputs) == len(sample_scores):
                # Build per-prompt grouping using input text equality
                n_repeat = max(1, int(self.config.actor_rollout_ref.rollout.val_kwargs.n))
                # Map each sample to a group key
                # Detect if reduced taxonomy is active (matches training classifier name)
                _gdro_cls_name = str(self.config.algorithm.get("gdro_prompt_classifier", "")).lower()
                _use_reduced_tax = _gdro_cls_name in {"math_reduced1", "math-reduced1", "math_reduced_1"}

                def to_group_key(ei: dict | None, prompt_text: str) -> str:
                    if isinstance(ei, dict):
                        lvl = ei.get("level", None)
                        typ = ei.get("type", None)
                        if lvl is not None or typ is not None:
                            try:
                                # Normalize Level string like "Level 3" -> 3
                                if isinstance(lvl, str) and lvl.lower().startswith("level"):
                                    lvl_num = int(str(lvl).split()[-1])
                                else:
                                    lvl_num = int(lvl) if lvl is not None else None
                            except Exception:
                                lvl_num = None
                            t = str(typ) if typ is not None else "unknown"
                            t_norm = "".join(ch for ch in t if ch.isalnum())  # remove spaces and symbols
                            # Apply coarsening only when reduced taxonomy is enabled
                            if _use_reduced_tax:
                                t_lower = t_norm.lower()
                                if t_lower in {"precalculus", "prealgebra", "prealgebraandprecalculus"}:
                                    t_norm = "Precalc"
                                if isinstance(lvl_num, int) and lvl_num <= 2:
                                    lvl_num = 2
                            return f"{t_norm}_Level{lvl_num if lvl_num is not None else 'NA'}"
                    return "unknown_LevelNA"

                groups = []
                for idx in range(len(sample_scores)):
                    ei = sample_extra_infos[idx] if idx < len(sample_extra_infos) else None
                    groups.append(to_group_key(ei, sample_inputs[idx]))

                # Aggregate per prompt
                prompt_to_indices = defaultdict(list)
                for i, p in enumerate(sample_inputs):
                    prompt_to_indices[p].append(i)
                # Compute per-prompt pass@1 and pass@k (k = n_repeat)
                k = n_repeat
                prompt_group = {}
                prompt_pass1 = {}
                prompt_passk = {}
                # Minimal extra signals: optionally also compute k=2, k=4, and k=8 if available
                compute_k2 = (n_repeat >= 2)
                prompt_pass2 = {} if compute_k2 else None
                compute_k4 = (n_repeat >= 4)
                prompt_pass4 = {} if compute_k4 else None
                compute_k8 = (n_repeat >= 8)
                prompt_pass8 = {} if compute_k8 else None
                compute_k16 = (n_repeat >= 16)
                prompt_pass16 = {} if compute_k16 else None
                compute_k32 = (n_repeat >= 32)
                prompt_pass32 = {} if compute_k32 else None
                for p, idxs in prompt_to_indices.items():
                    # Determine group from first index
                    g = groups[idxs[0]] if idxs else "unknown_LevelNA"
                    prompt_group[p] = g
                    # pass@1: correctness of first sample
                    s1 = sample_scores[idxs[0]]
                    prompt_pass1[p] = 1.0 if (s1 is not None and float(s1) > 0.5) else 0.0
                    # pass@k: any correct among its indices
                    any_correct = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:k])
                    prompt_passk[p] = 1.0 if any_correct else 0.0
                    if compute_k2:
                        any2 = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:2])
                        prompt_pass2[p] = 1.0 if any2 else 0.0
                    if compute_k4:
                        any4 = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:4])
                        prompt_pass4[p] = 1.0 if any4 else 0.0
                    if compute_k8:
                        any8 = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:8])
                        prompt_pass8[p] = 1.0 if any8 else 0.0
                    if compute_k16:
                        any16 = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:16])
                        prompt_pass16[p] = 1.0 if any16 else 0.0
                    if compute_k32:
                        any32 = any((sample_scores[j] is not None and float(sample_scores[j]) > 0.5) for j in idxs[:32])
                        prompt_pass32[p] = 1.0 if any32 else 0.0

                # Overall dataset-level pass metric: only keep highest k (k = n_repeat)
                if len(prompt_passk) > 0:
                    overall_passk = float(sum(prompt_passk.values()) / len(prompt_passk))
                    metric_dict[f"val-core/acc/pass@{k}"] = overall_passk

                # Reduce per-group
                group_to_pass1 = defaultdict(list)
                group_to_passk = defaultdict(list)
                for p, g in prompt_group.items():
                    group_to_pass1[g].append(prompt_pass1[p])
                    group_to_passk[g].append(prompt_passk[p])
                    # no @4 logic

                if len(group_to_pass1) > 0:
                    # Compute stats and log compact summaries
                    means1 = {g: float(np.mean(v)) for g, v in group_to_pass1.items() if len(v) > 0}
                    meansk = {g: float(np.mean(v)) for g, v in group_to_passk.items() if len(v) > 0}
                    std1 = {g: float(np.std(v)) for g, v in group_to_pass1.items() if len(v) > 1}
                    stdk = {g: float(np.std(v)) for g, v in group_to_passk.items() if len(v) > 1}
                    support = {g: int(len(v)) for g, v in group_to_passk.items()}

                    if len(meansk) > 0:
                        # Emit only group metrics at k = n_repeat
                        worst_k = min(meansk.values())
                        mean_k = float(np.mean(list(meansk.values())))
                        metric_dict[f"val-group/worst@{k}/mean"] = worst_k
                        metric_dict[f"val-group/mean@{k}/mean"] = mean_k

                    # Skip per-group and support logging to minimize overhead
                    # --------------------------------------------------------
                    # Optional: evaluation over dynamic pass@k accuracy bins
                    # --------------------------------------------------------
                    try:
                        # Compute over dynamic pass@k accuracy bins if available.
                        # For GRPO baselines (no self.gdro), optionally instantiate a temporary eval-only classifier
                        # when algorithm.gdro_prompt_classifier requests passk-based bins.
                        gdro_has_classifier = hasattr(self, "gdro") and hasattr(self.gdro, "classifier")
                        uids_arr = reward_extra_infos_dict.get("uid", None)
                        classifier_obj = None
                        if gdro_has_classifier:
                            classifier_obj = self.gdro.classifier
                        elif uids_arr is not None:
                            # Eval-only temporary classifier for GRPO when a passk-based classifier is requested
                            try:
                                gdro_cls_name = str(self.config.algorithm.get("gdro_prompt_classifier", "")).lower()
                                if gdro_cls_name in {"passk_online", "acc_online", "online_passk"}:
                                    from verl.trainer.group_dro_grpo import ClassDroExp3p, Exp3pConfig
                                    temp_cfg = Exp3pConfig()
                                    temp_cfg.prompt_classifier = gdro_cls_name
                                    # Carry over passk knobs if provided
                                    if hasattr(self.config.algorithm, "passk_num_bins"):
                                        temp_cfg.passk_num_bins = int(self.config.algorithm.passk_num_bins)
                                    if hasattr(self.config.algorithm, "passk_history_len"):
                                        temp_cfg.passk_history_len = int(self.config.algorithm.passk_history_len)
                                    if hasattr(self.config.algorithm, "passk_hysteresis"):
                                        temp_cfg.passk_hysteresis = float(self.config.algorithm.passk_hysteresis)
                                    if hasattr(self.config.algorithm, "passk_edges"):
                                        temp_cfg.passk_edges = str(self.config.algorithm.passk_edges)
                                    if hasattr(self.config.algorithm, "passk_exclude_extremes"):
                                        temp_cfg.passk_exclude_extremes = bool(self.config.algorithm.passk_exclude_extremes)
                                    temp_gdro = ClassDroExp3p(temp_cfg)
                                    # Update running uid accuracies using current step's per-prompt pass@k
                                    uids_list = uids_arr.tolist() if hasattr(uids_arr, "tolist") else list(uids_arr)
                                    for p, idxs in prompt_to_indices.items():
                                        if not idxs:
                                            continue
                                        first_idx = idxs[0]
                                        if 0 <= first_idx < len(uids_list):
                                            uid_val = str(uids_list[first_idx])
                                            ok = float(prompt_passk.get(p, 0.0))
                                            temp_gdro.update_with_passk([uid_val], [ok])
                                    classifier_obj = temp_gdro.classifier
                            except Exception:
                                classifier_obj = None
                        if classifier_obj is not None and uids_arr is not None:
                            uids_list = uids_arr.tolist() if hasattr(uids_arr, "tolist") else list(uids_arr)
                            prompt_accbin = {}
                            for p, idxs in prompt_to_indices.items():
                                if not idxs:
                                    continue
                                first_idx = idxs[0]
                                uid_val = None
                                if 0 <= first_idx < len(uids_list):
                                    uid_val = str(uids_list[first_idx])
                                md = {"uid": uid_val} if uid_val is not None else None
                                try:
                                    bin_label = classifier_obj.classify(sample_inputs[first_idx], md)
                                except Exception:
                                    bin_label = "accbin_unk"
                                prompt_accbin[p] = bin_label

                            # Reduce per-accbin for pass@k only (k = n_repeat)
                            from collections import defaultdict as _dd
                            accbin_to_passk = _dd(list)

                            for p, bin_label in prompt_accbin.items():
                                if p in prompt_passk:
                                    accbin_to_passk[bin_label].append(prompt_passk[p])

                            # Emit compact accbin-based metrics for current k
                            if len(accbin_to_passk) > 0:
                                meansk_ab = {g: float(np.mean(v)) for g, v in accbin_to_passk.items() if len(v) > 0}
                                if len(meansk_ab) > 0:
                                    worst_accbin_k = min(meansk_ab.values())
                                    metric_dict[f"val-group/mean_accbins@{k}/mean"] = float(np.mean(list(meansk_ab.values())))
                                    metric_dict[f"val-group/worst_accbins@{k}/mean"] = worst_accbin_k
                    except Exception:
                        pass
        except Exception:
            pass

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
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

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

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

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                # Prefer stable per-prompt identifiers from dataset 'index' if available; fallback to UUID4
                try:
                    self._rollout_step_state = {}
                    idx_arr = batch.non_tensor_batch.get("index", None)
                    if idx_arr is not None:
                        idx_list = idx_arr.tolist() if hasattr(idx_arr, "tolist") else list(idx_arr)
                        batch.non_tensor_batch["uid"] = np.array(
                            [f"idx:{str(x)}" for x in idx_list], dtype=object
                        )
                    else:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                except Exception:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                # Rollout-GDRO v3: Rollout allocation using PREVIOUS step's classification
                # At step N, use step N-1's classification to allocate rollouts
                # After generation, we'll classify and store for step N+1
                selected_indices = None
                is_budget_allocated = False
                rollout_class_ids: list[str] | None = None
                rollout_budgets: np.ndarray | None = None
                rollout_expanded_class_ids: list[str] | None = None
                rollout_class_weight_map: dict | None = None
                using_prev_step_allocation = False
                
                if self.rollout_budget_mode == "groupdro" and self.rollout_allocator is not None:
                    allocator = self.rollout_allocator
                    num_prompts = len(batch)
                    
                    # Check if we have previous step's classification to use
                    if (
                        self._rollout_prev_step_class_ids is not None
                        and len(self._rollout_prev_step_class_ids) == num_prompts
                    ):
                        # Use previous step's classification for allocation
                        try:
                            rollout_class_ids = self._rollout_prev_step_class_ids
                            if self._rollout_prev_step_weights is not None:
                                weights = self._rollout_prev_step_weights
                            else:
                                weights = self.rollout_allocator.weights_for_samples(rollout_class_ids)
                            rollout_class_weight_map = self._rollout_prev_step_weight_map
                            
                            # Allocate rollouts based on previous step's weights
                            total_budget = num_prompts * int(self.config.actor_rollout_ref.rollout.n)
                            # Rollout-GDRO (Problem 2.1): choose discrete rollout counts per bin with budget constraint
                            if (
                                self.rollout_prob21_enable
                                and hasattr(allocator, "rollout_choose_arms")
                            ):
                                class_counts = Counter(rollout_class_ids)
                                bin_n_map, chosen_sum = allocator.rollout_choose_arms(
                                    class_counts, total_budget, self.rollout_budget_n_min
                                )
                                if bin_n_map is not None and chosen_sum == total_budget:
                                    budgets = np.array(
                                        [int(bin_n_map.get(cid, self.rollout_budget_n_min)) for cid in rollout_class_ids],
                                        dtype=int,
                                    )
                                    if budgets.sum() == total_budget:
                                        selected_indices = []
                                        for idx, cnt in enumerate(budgets.tolist()):
                                            if cnt > 0:
                                                selected_indices.extend([idx] * int(cnt))
                                        rollout_budgets = budgets
                                        rollout_expanded_class_ids = [rollout_class_ids[idx] for idx in selected_indices]
                                        self._rollout_step_state["bin_n_map"] = bin_n_map
                                        self._rollout_step_state["class_counts"] = class_counts
                                        using_prev_step_allocation = True
                            # Fallback to proportional allocator
                            if not using_prev_step_allocation:
                                rollout_budgets, selected_indices, rollout_expanded_class_ids = self._allocate_rollouts_by_group(
                                    rollout_class_ids, weights, total_budget
                                )
                                using_prev_step_allocation = True
                        except Exception as exc:
                            warnings.warn(
                                f"[rollout_budget] Falling back to uniform allocation due to error: {exc}",
                                stacklevel=2,
                            )
                            self._rollout_step_state = {}
                            self._rollout_allocator_fallbacks += 1
                            rollout_class_ids = None
                            rollout_budgets = None
                            selected_indices = None
                    else:
                        # First step or batch size changed: uniform allocation (warmup)
                        # Classification will happen AFTER generation for next step
                        self._rollout_allocator_fallbacks += 1
                
                # Now create gen_batch (this modifies batch.non_tensor_batch)
                gen_batch = self._get_gen_batch(batch)
                
                # Apply rollout allocation if successful
                if self.rollout_budget_mode == "groupdro" and self.rollout_allocator is not None:
                    class_ids = rollout_class_ids
                    rollout_metrics_logged = False
                    allocator = self.rollout_allocator
                    
                    if rollout_budgets is not None and selected_indices is not None and rollout_expanded_class_ids is not None:
                        budgets = rollout_budgets
                        expanded_class_ids = rollout_expanded_class_ids
                        class_weight_map = rollout_class_weight_map
                        
                        # Expand gen_batch using selected_indices
                        gen_batch = gen_batch.select_idxs(selected_indices)
                        
                        # Store state for update
                        self._rollout_step_state = {
                            "class_ids": class_ids,
                            "expanded_class_ids": expanded_class_ids,
                            "selected_indices": selected_indices,
                        }

                        is_budget_allocated = True
                        
                        # Log metrics
                        total_budget = len(class_ids) * int(self.config.actor_rollout_ref.rollout.n)
                        budgets_np = budgets.astype(float)
                        total = float(budgets_np.sum()) if budgets_np.size > 0 else 0.0
                        if total > 0:
                            probs = budgets_np / total
                            entropy = float(-(probs * np.log(probs + 1e-8)).sum())
                        else:
                            entropy = 0.0
                        
                        unique_classes = len(set(class_ids))
                        base_budget = float(self.rollout_budget_n_min * len(class_ids))
                        total_extra = float(max(0, total_budget - base_budget))
                        n_min_hits = float((budgets_np <= (self.rollout_budget_n_min + 1e-6)).sum()) if budgets_np.size > 0 else 0.0
                        n_max_hits = float((budgets_np >= (self.rollout_budget_n_max - 1e-6)).sum()) if budgets_np.size > 0 else 0.0
                        metrics.update({
                            "rollout_alloc/min": float(budgets_np.min()) if budgets_np.size > 0 else 0.0,
                            "rollout_alloc/max": float(budgets_np.max()) if budgets_np.size > 0 else 0.0,
                            "rollout_alloc/mean": float(budgets_np.mean()) if budgets_np.size > 0 else 0.0,
                            "rollout_alloc/std": float(budgets_np.std()) if budgets_np.size > 0 else 0.0,
                            "rollout_alloc/entropy": entropy,
                            "rollout_alloc/num_prompts": float(len(class_ids)),
                            "rollout_alloc/n_min": float(self.rollout_budget_n_min),
                            "rollout_alloc/n_max": float(self.rollout_budget_n_max),
                            "rollout_alloc/total_budget": float(total_budget),
                            "rollout_alloc/base_budget": base_budget,
                            "rollout_alloc/extra_budget": total_extra,
                            "rollout_alloc/num_classes": float(unique_classes),
                            "rollout_alloc/passk_shared": 1.0 if self._rollout_allocator_shared_passk else 0.0,
                            "rollout_alloc/n_min_hits": n_min_hits,
                            "rollout_alloc/n_max_hits": n_max_hits,
                            "rollout_alloc/n_min_hit_frac": float(n_min_hits / max(1.0, len(class_ids))),
                            "rollout_alloc/n_max_hit_frac": float(n_max_hits / max(1.0, len(class_ids))),
                            "rollout_alloc/using_prev_step": 1.0 if using_prev_step_allocation else 0.0,
                        })
                        rollout_metrics_logged = True
                        
                        classifier = getattr(allocator, "classifier", None)
                        if classifier is not None:
                            try:
                                metrics["rollout_alloc/passk_history_len"] = float(getattr(classifier, "history_len", 0))
                                metrics["rollout_alloc/passk_num_bins"] = float(getattr(classifier, "num_bins", 0))
                            except Exception:
                                pass
                        
                        if class_weight_map:
                            top_weight_items = sorted(class_weight_map.items(), key=lambda kv: kv[1], reverse=True)[:3]
                            for rank, (cid, w_val) in enumerate(top_weight_items, start=1):
                                metrics[f"rollout_alloc/weight_top{rank}"] = float(w_val)
                        
                        counts = Counter(class_ids)
                        # Track unknown/unk class share (similar to gdro/class_count/unknown_share)
                        total_prompts = len(class_ids)
                        unk_count = sum(v for k, v in counts.items() if "unk" in str(k).lower())
                        metrics["rollout_alloc/unknown_count"] = float(unk_count)
                        metrics["rollout_alloc/unknown_share"] = float(unk_count / max(1, total_prompts))
                        
                        per_class_budget: Dict[str, int] = defaultdict(int)
                        for cid, budget_val in zip(class_ids, budgets.tolist()):
                            per_class_budget[cid] += int(budget_val)
                        for cid, total_val in per_class_budget.items():
                            metrics[f"rollout_alloc/class_total/{cid}"] = float(total_val)
                            if total > 0:
                                metrics[f"rollout_alloc/class_share/{cid}"] = float(total_val / total)
                            denom = max(1, counts.get(cid, 0))
                            metrics[f"rollout_alloc/class_mean/{cid}"] = float(total_val / denom)
                    else:
                        self._rollout_step_state = {}

                    if not rollout_metrics_logged:
                        fallback_prompts = float(len(class_ids) if class_ids is not None else len(batch))
                        total_budget = float(fallback_prompts * int(self.config.actor_rollout_ref.rollout.n))
                        base_budget = float(self.rollout_budget_n_min * fallback_prompts)
                        total_extra = float(max(0.0, total_budget - base_budget))
                        metrics.update(
                            {
                                "rollout_alloc/min": float(self.rollout_budget_n_min),
                                "rollout_alloc/max": float(self.rollout_budget_n_max),
                                "rollout_alloc/mean": float(self.rollout_budget_n_min),
                                "rollout_alloc/std": 0.0,
                                "rollout_alloc/entropy": 0.0,
                                "rollout_alloc/num_prompts": fallback_prompts,
                                "rollout_alloc/n_min": float(self.rollout_budget_n_min),
                                "rollout_alloc/n_max": float(self.rollout_budget_n_max),
                                "rollout_alloc/total_budget": total_budget,
                                "rollout_alloc/base_budget": base_budget,
                                "rollout_alloc/extra_budget": total_extra,
                                "rollout_alloc/num_classes": float(len(set(class_ids)) if class_ids else 0.0),
                                "rollout_alloc/passk_shared": 1.0 if self._rollout_allocator_shared_passk else 0.0,
                                "rollout_alloc/n_min_hits": fallback_prompts,
                                "rollout_alloc/n_max_hits": 0.0,
                                "rollout_alloc/n_min_hit_frac": 1.0 if fallback_prompts > 0 else 0.0,
                                "rollout_alloc/n_max_hit_frac": 0.0,
                            }
                        )

                metrics["rollout_alloc/fallback_count"] = float(self._rollout_allocator_fallbacks)

                # Optional rollout budget allocation (vanilla / knapsack / knapsack_group_dro)
                ba_cfg = None
                try:
                    ba_cfg = getattr(self.config.trainer, "rollout_budget_allocation", None)
                except Exception:
                    ba_cfg = None
                if ba_cfg is None:
                    try:
                        ba_cfg = getattr(self.config.algorithm, "rollout_budget_allocation", None)
                    except Exception:
                        ba_cfg = None

                if ba_cfg is not None:
                    try:
                        method = str(getattr(ba_cfg, "method", "")).lower()
                    except Exception:
                        method = ""
                else:
                    method = ""

                if (
                    selected_indices is None
                    and method in {"vanilla", "knapsack", "knapsack_group_dro", "knapsack_gdro_passk"}
                ):
                    total_budget = int(len(batch)) * int(self.config.actor_rollout_ref.rollout.n)
                    if method == "vanilla":
                        gen_batch, budgets = budget_allocation_vanilla(gen_batch, total_budget)
                    elif method == "knapsack":
                        score_key = str(getattr(ba_cfg, "score_key", "status"))
                        gen_batch, budgets = budget_allocation_knapsack(gen_batch, total_budget, score_key=score_key)
                    elif method == "knapsack_group_dro":
                        score_key = str(getattr(ba_cfg, "score_key", "status"))
                        category_key = getattr(ba_cfg, "category_key", None)
                        eta_q = float(getattr(ba_cfg, "eta_q", 0.10))
                        gamma = float(getattr(ba_cfg, "gamma", 0.10))
                        ema_alpha = float(getattr(ba_cfg, "ema_alpha", 0.15))
                        gen_batch, budgets = budget_allocation_knapsack_group_dro(
                            gen_batch,
                            total_budget,
                            score_key=score_key,
                            category_key=category_key,
                            eta_q=eta_q,
                            gamma=gamma,
                            ema_alpha=ema_alpha,
                        )
                    else:  # knapsack_gdro_passk
                        eta_q = float(getattr(ba_cfg, "eta_q", 0.10))
                        gamma = float(getattr(ba_cfg, "gamma", 0.10))
                        ema_alpha = float(getattr(ba_cfg, "ema_alpha", 0.15))
                        # Optional passk_edges from algorithm config
                        edges_str = getattr(ba_cfg, "passk_edges", None)
                        passk_edges = None
                        if isinstance(edges_str, str) and edges_str:
                            try:
                                passk_edges = [float(x) for x in edges_str.split(",")]
                            except Exception:
                                passk_edges = None
                        focus_enable = bool(getattr(ba_cfg, "passk_focus_enable", False))
                        focus_min = float(getattr(ba_cfg, "passk_focus_min", 0.05))
                        gen_batch, budgets = budget_allocation_knapsack_gdro_passk(
                            gen_batch,
                            total_budget,
                            passk_edges=passk_edges,
                            eta_q=eta_q,
                            gamma=gamma,
                            ema_alpha=ema_alpha,
                            passk_focus_enable=focus_enable,
                            passk_focus_min=focus_min,
                        )

                    # Rebuild indices so that original batch aligns with generated prompts
                    selected_indices = []
                    for tid, tb in enumerate(budgets.tolist()):
                        if tb > 0:
                            selected_indices.extend([tid] * int(tb))
                    is_budget_allocated = True

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                if not is_budget_allocated:
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        # Ensure knapsack metrics survive rollout: copy from request if missing
                        try:
                            if (
                                hasattr(gen_batch, "meta_info")
                                and isinstance(gen_batch.meta_info, dict)
                                and "knapsack_metrics" in gen_batch.meta_info
                                and hasattr(gen_batch_output, "meta_info")
                                and isinstance(gen_batch_output.meta_info, dict)
                                and "knapsack_metrics" not in gen_batch_output.meta_info
                            ):
                                gen_batch_output.meta_info["knapsack_metrics"] = gen_batch.meta_info["knapsack_metrics"]
                        except Exception:
                            pass

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
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

                    # Align base batch to generated prompts
                    if is_budget_allocated and selected_indices is not None:
                        batch = batch.select_idxs(selected_indices)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Log knapsack resource-sharing metrics if available
                    km = gen_batch_output.meta_info.get("knapsack_metrics") if hasattr(gen_batch_output, "meta_info") else None
                    if isinstance(km, dict):
                        try:
                            knapsack_metrics = {f"knapsack/{k}": v for k, v in km.items()}
                            metrics.update(knapsack_metrics)
                        except Exception:
                            pass

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            clip_ratio = getattr(self.config.actor_rollout_ref.actor, "clip_ratio", None)
                            clip_high = getattr(self.config.actor_rollout_ref.actor, "clip_ratio_high", None)
                            clip_low = getattr(self.config.actor_rollout_ref.actor, "clip_ratio_low", None)
                            if clip_high is None:
                                clip_high = clip_ratio
                            if clip_low is None:
                                clip_low = clip_ratio
                            if clip_high is not None:
                                batch.meta_info["clip_ratio_high"] = float(clip_high)
                            if clip_low is not None:
                                batch.meta_info["clip_ratio_low"] = float(clip_low)
                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Update online pass@k EMA per UID so allocation can use 1 - pass@k
                            try:
                                from verl.utils.budget_allocation import update_online_passk
                                uids_arr = reward_extra_infos_dict.get("uid", [])
                                acc_arr = reward_extra_infos_dict.get("acc", [])
                                if len(uids_arr) == len(acc_arr) and len(uids_arr) > 0:
                                    uids_list = [str(u) for u in (uids_arr.tolist() if hasattr(uids_arr, "tolist") else uids_arr)]
                                    # any_correct per UID (pass@k signal)
                                    any_correct = {}
                                    for uid, a in zip(uids_list, acc_arr, strict=True):
                                        any_correct[uid] = max(float(a) > 0.0, any_correct.get(uid, 0.0))
                                    update_online_passk(list(any_correct.keys()), [1.0 if v else 0.0 for v in any_correct.values()])
                            except Exception:
                                pass

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            rewards = batch.batch["token_level_scores"]
                            batch.batch["token_level_rewards"] = rewards

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # Before compute_advantage: attach per-sample class weights if Problem 1 enabled
                        # Problem 2 v3 / prob2.1: classify AFTER generation and store for next step's allocation
                        class_ids = None
                        prob3_class_ids: Optional[List[str]] = None
                        prompt_texts: Optional[List[str]] = None
                        metadatas: Optional[List[dict]] = None
                        if self.gdro_enabled and self.gdro is not None:
                            # Use unified metadata preparation (same as Problem 2)
                            prompt_texts, metadatas = self._prepare_classification_inputs(batch)
                            
                            # Choose IDs based on weight mode
                            if self.gdro_weight_mode == "prompt":
                                # Use stable per-sample UIDs
                                uids = batch.non_tensor_batch.get("uid", None)
                                if uids is None:
                                    import uuid as _uuid
                                    batch.non_tensor_batch["uid"] = np.array(
                                        [str(_uuid.uuid4()) for _ in range(len(batch))], dtype=object
                                    )
                                    uids = batch.non_tensor_batch["uid"]
                                ids = [str(u) for u in uids.tolist()]
                                if self.offpolicy_grpo_enable:
                                    try:
                                        prob3_class_ids = self.gdro.classify_batch(prompt_texts, metadatas)
                                    except Exception:
                                        prob3_class_ids = None
                            else:
                                # Always classify AFTER generation for consistent Problem 1 & Problem 2 v3
                                class_ids = self.gdro.classify_batch(prompt_texts, metadatas)
                                ids = class_ids
                                
                                # Store expanded_class_ids for rollout_step_state (for update_with_losses)
                                self._rollout_step_state["expanded_class_ids"] = class_ids
                                
                                # Problem 2 v3: Store classification for NEXT step's rollout allocation
                                # Convert expanded batch class_ids back to per-prompt class_ids
                                # The expanded batch has n rollouts per prompt, we need per-prompt class_ids
                                n_rollouts = int(self.config.actor_rollout_ref.rollout.n)
                                num_expanded = len(class_ids)
                                num_prompts = num_expanded // n_rollouts if n_rollouts > 0 else num_expanded
                                
                                if num_prompts > 0 and num_expanded == num_prompts * n_rollouts:
                                    # Extract per-prompt class_ids (take first of each group)
                                    per_prompt_class_ids = [class_ids[i * n_rollouts] for i in range(num_prompts)]
                                    # Compute weights for next step's allocation
                                    per_prompt_weights = self.rollout_allocator.weights_for_samples(per_prompt_class_ids) if self.rollout_allocator else None
                                    per_prompt_weight_map = self.rollout_allocator.compute_weights(per_prompt_class_ids) if self.rollout_allocator else None
                                    
                                    # Store for next step
                                    self._rollout_prev_step_class_ids = per_prompt_class_ids
                                    self._rollout_prev_step_weights = per_prompt_weights
                                    self._rollout_prev_step_weight_map = per_prompt_weight_map
                                    
                                metrics["gdro/reused_rollout_class_ids"] = 0.0  # v3: always fresh classification
                            # Compute per-sample weights
                            w_vec = self.gdro.weights_for_samples(ids)
                            # Normalize weights to mean 1.0 to avoid shrinking gradients
                            if self.gdro_weight_mode in ("prompt", "class"):
                                s = float(w_vec.sum().item())
                                n = max(1.0, float(w_vec.numel()))
                                if s > 0:
                                    w_vec = w_vec * (n / s)
                                # Clamp to configured max
                                try:
                                    max_w = float(self.gdro.cfg.max_class_weight)
                                    w_vec = torch.clamp(w_vec, max=max_w)
                                except Exception:
                                    pass
                            # device/dtype alignment later in actor; keep as CPU float32 here
                            if self.gdro_apply_weights:
                                batch.batch["gdro_class_weight"] = w_vec
                            # GDRO metrics (pre-adv): class-wise weight entropy & counts
                            class_weight_map = self.gdro.compute_weights(ids)
                            if len(class_weight_map) > 0:
                                vals = torch.tensor(list(class_weight_map.values()), dtype=torch.float32)
                                probs = vals / (vals.sum() + 1e-8)
                                weight_entropy = float(-(probs * (probs + 1e-8).log()).sum().item())
                                metrics["gdro/weight_entropy"] = weight_entropy
                                if self.gdro_weight_mode == "prompt":
                                    metrics["gdro/num_prompts_weighted"] = len(class_weight_map)
                                else:
                                    metrics["gdro/num_classes"] = len(class_weight_map)
                                # Log per-accbin masked weights (post-focus, pre-renorm outside EXP3P)
                                # This complements top_weight which only shows top-5.
                                try:
                                    for cid, wv in class_weight_map.items():
                                        if isinstance(cid, str) and cid.startswith("accbin_"):
                                            metrics[f"gdro/masked_weight/{cid}"] = float(wv)
                                except Exception:
                                    pass
                                # Log top-5 by current weight (helps verify high-weight == hard)
                                try:
                                    top5w = sorted(class_weight_map.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for cid, wv in top5w:
                                        metrics[f"gdro/top_weight/{cid}"] = float(wv)
                                except Exception:
                                    pass
                            # Per-sample weight stats
                            try:
                                metrics["gdro/weight_mean"] = float(w_vec.mean().item())
                                metrics["gdro/weight_min"] = float(w_vec.min().item())
                                metrics["gdro/weight_max"] = float(w_vec.max().item())
                            except Exception:
                                pass
                            # Unknown class share
                            if self.gdro_weight_mode != "prompt":
                                try:
                                    cnt = Counter(ids)
                                    total = sum(cnt.values()) or 1
                                    unknown_cnt = sum(v for k, v in cnt.items() if str(k).lower().startswith("unknown"))
                                    metrics["gdro/class_count/unknown"] = int(unknown_cnt)
                                    metrics["gdro/class_count/unknown_share"] = float(unknown_cnt / total)
                                except Exception:
                                    pass
                            if class_ids is not None:
                                prob3_class_ids = class_ids

                        # Prob2.1-only runs still need classification for rollout allocation even when GDRO is disabled
                        rollout_expanded_class_ids: Optional[List[str]] = None
                        if (
                            self.rollout_prob21_enable
                            and not self.gdro_enabled
                            and self.rollout_budget_mode == "groupdro"
                            and self.rollout_allocator is not None
                        ):
                            if prompt_texts is None or metadatas is None:
                                prompt_texts, metadatas = self._prepare_classification_inputs(batch)
                            try:
                                rollout_expanded_class_ids = self.rollout_allocator.classify_batch(prompt_texts, metadatas)
                            except Exception:
                                rollout_expanded_class_ids = None
                            if rollout_expanded_class_ids is not None:
                                n_rollouts = int(self.config.actor_rollout_ref.rollout.n)
                                num_expanded = len(rollout_expanded_class_ids)
                                num_prompts = num_expanded // n_rollouts if n_rollouts > 0 else num_expanded
                                if num_prompts > 0 and num_expanded == num_prompts * n_rollouts:
                                    per_prompt_class_ids = [
                                        rollout_expanded_class_ids[i * n_rollouts] for i in range(num_prompts)
                                    ]
                                    try:
                                        per_prompt_weights = self.rollout_allocator.weights_for_samples(per_prompt_class_ids)
                                        per_prompt_weight_map = self.rollout_allocator.compute_weights(per_prompt_class_ids)
                                    except Exception:
                                        per_prompt_weights = None
                                        per_prompt_weight_map = None
                                    self._rollout_prev_step_class_ids = per_prompt_class_ids
                                    self._rollout_prev_step_weights = per_prompt_weights
                                    self._rollout_prev_step_weight_map = per_prompt_weight_map
                                prob3_class_ids = rollout_expanded_class_ids

                        if self.offpolicy_grpo_enable:
                            if prob3_class_ids is None:
                                if prompt_texts is None or metadatas is None:
                                    prompt_texts, metadatas = self._prepare_classification_inputs(batch)
                                classifier = self.offpolicy_classifier
                                if classifier is not None:
                                    try:
                                        prob3_class_ids = classifier.classify_batch(prompt_texts, metadatas)
                                    except Exception:
                                        prob3_class_ids = None
                            self._update_prob3_state(prob3_class_ids, metrics)
                        else:
                            self._reset_prob3_state()

                        self._apply_prob3_behavior_log_probs(batch, metrics)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # Additional GDRO batch-level stats (reward/accuracy by class) for GSM8K/MATH rule-based (Problem 1)
                        if self.gdro_enabled and self.gdro is not None and self.gdro_weight_mode != "prompt" and "token_level_rewards" in batch.batch:
                            try:
                                seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu()
                                # group by class and compute mean reward, log top-3 by frequency
                                counts = Counter(class_ids)
                                top_classes = [c for c, _ in counts.most_common(5)]
                                reward_by_class = defaultdict(list)
                                for idx, cid in enumerate(class_ids):
                                    reward_by_class[cid].append(float(seq_rewards[idx].item()))
                                for cid in top_classes:
                                    vals = reward_by_class.get(cid, None)
                                    if vals:
                                        metrics[f"gdro/class_reward_mean/{cid}"] = float(sum(vals) / max(1, len(vals)))
                                    metrics[f"gdro/class_count/{cid}"] = int(counts.get(cid, 0))
                                # Also log reward mean for top-5 by current weight if available
                                try:
                                    cw = self.gdro.compute_weights(class_ids)
                                    top5w = sorted(cw.items(), key=lambda x: x[1], reverse=True)[:5]
                                    for cid, _ in top5w:
                                        vals = reward_by_class.get(cid, None)
                                        if vals:
                                            metrics[f"gdro/top_weight_reward_mean/{cid}"] = float(sum(vals) / max(1, len(vals)))
                                except Exception:
                                    pass
                            except Exception:
                                pass

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # Update GDRO EXP3P cumulatives with true per-prompt GRPO losses (Problem 1)
                        if self.gdro_enabled and self.gdro is not None:
                            # Update schedules if configured
                            try:
                                step = int(self.global_steps)
                                # eta_q schedule
                                s_eta = self._gdro_sched.get("eta_q", {}) if hasattr(self, "_gdro_sched") else {}
                                t0 = int(s_eta.get("t0", 0)); t1 = int(s_eta.get("t1", 0))
                                if t1 > t0 and step >= t0:
                                    alpha = max(0.0, min(1.0, (step - t0) / max(1, (t1 - t0))))
                                    val = float(s_eta.get("start", self.gdro.cfg.eta_q)) * (1 - alpha) + float(s_eta.get("end", self.gdro.cfg.eta_q)) * alpha
                                    self.gdro.cfg.eta_q = float(val)
                                # gamma schedule
                                s_gam = self._gdro_sched.get("gamma", {}) if hasattr(self, "_gdro_sched") else {}
                                gt0 = int(s_gam.get("t0", 0)); gt1 = int(s_gam.get("t1", 0))
                                if gt1 > gt0 and step >= gt0:
                                    alpha_g = max(0.0, min(1.0, (step - gt0) / max(1, (gt1 - gt0))))
                                    gval = float(s_gam.get("start", self.gdro.cfg.gamma)) * (1 - alpha_g) + float(s_gam.get("end", self.gdro.cfg.gamma)) * alpha_g
                                    self.gdro.cfg.gamma = float(gval)
                            except Exception:
                                pass
                            per_sample_lb = actor_output.meta_info.get("per_sample_pg_loss", None)
                            if per_sample_lb is not None:
                                if self.gdro_weight_mode == "prompt":
                                    uids = batch.non_tensor_batch.get("uid", None)
                                    if uids is not None:
                                        ids = [str(u) for u in uids.tolist()]
                                        self.gdro.update_with_losses(ids, per_sample_lb)
                                else:
                                    if 'class_ids' in locals() and class_ids is not None:
                                        self.gdro.update_with_losses(class_ids, per_sample_lb)
                                # Optional: log top cumulatives only for class mode to keep logs concise
                                if self.gdro_weight_mode != "prompt":
                                    try:
                                        cum = self.gdro.cumulative_class_scores
                                        if len(cum) > 0:
                                            top5 = sorted(cum.items(), key=lambda x: x[1], reverse=True)[:5]
                                            for cid, val in top5:
                                                metrics[f"gdro/cum_score/{cid}"] = float(val)
                                    except Exception:
                                        pass

                            # Problem 2: Update exp3p state with expanded batch losses
                            if (
                                self.rollout_budget_mode == "groupdro"
                                and self.rollout_allocator is not None
                                and per_sample_lb is not None
                            ):
                                expanded_ids = self._rollout_step_state.get("expanded_class_ids")
                                if expanded_ids is not None and len(expanded_ids) == per_sample_lb.shape[0]:
                                    try:
                                        # Update using expanded class_ids (shared exp3p when available)
                                        self.rollout_allocator.update_with_losses(expanded_ids, per_sample_lb)
                                    except Exception:
                                        pass
                                if self.rollout_prob21_enable:
                                    bin_n_map = self._rollout_step_state.get("bin_n_map")
                                    class_counts = self._rollout_step_state.get("class_counts")
                                    if bin_n_map is not None and expanded_ids is not None:
                                        try:
                                            bar_n = float(self.config.actor_rollout_ref.rollout.n)
                                            self.rollout_allocator.update_rollout_arm_losses(
                                                bin_n_map,
                                                expanded_ids,
                                                per_sample_lb,
                                                bar_n,
                                                self.rollout_budget_dual_mu,
                                            )
                                            total_prompts = float(sum(class_counts.values())) if class_counts else 0.0
                                            realized_mean = 0.0
                                            if total_prompts > 0:
                                                realized_mean = float(
                                                    sum(
                                                        int(bin_n_map.get(c, self.rollout_budget_n_min))
                                                        * int(class_counts.get(c, 0))
                                                        for c in bin_n_map
                                                    )
                                                ) / max(1.0, total_prompts)
                                            self.rollout_budget_dual_mu += float(self.rollout_budget_dual_lr) * (
                                                realized_mean - bar_n
                                            )
                                            metrics["rollout_alloc/dual_mu"] = float(self.rollout_budget_dual_mu)
                                        except Exception:
                                            pass

                            if self.offpolicy_grpo_enable and per_sample_lb is not None:
                                self._update_offpolicy_weights(per_sample_lb, metrics, batch)

                            # Online pass@k update for dynamic accuracy buckets
                            try:
                                # Build per-uid pass@k using either explicit correctness or reward threshold
                                uids_arr = batch.non_tensor_batch.get("uid", None)
                                if uids_arr is not None:
                                    uids_list = uids_arr.tolist() if hasattr(uids_arr, "tolist") else list(uids_arr)
                                    # correctness per sample (per response)
                                    corr_arr = batch.non_tensor_batch.get("is_correct", None)
                                    per_resp_correct: list[float] | None = None
                                    if corr_arr is not None:
                                        per_resp_correct = [1.0 if (bool(x) if not isinstance(x, (list, np.ndarray)) else bool(x[0])) else 0.0 for x in corr_arr]
                                    else:
                                        # fallback: sequence reward > 0.5 indicates correct
                                        if "token_level_rewards" in batch.batch:
                                            seq = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().tolist()
                                        else:
                                            seq = batch.batch.get("token_level_scores", torch.zeros(len(uids_list))).sum(dim=-1).detach().cpu().tolist()
                                        per_resp_correct = [1.0 if float(s) > 0.5 else 0.0 for s in seq]

                                    # aggregate to per-uid (original prompt) using any-of-k
                                    uid2any: dict[str, float] = {}
                                    for idx, uid in enumerate(uids_list):
                                        key = str(uid)
                                        ok = float(per_resp_correct[idx]) > 0.5
                                        if key not in uid2any:
                                            uid2any[key] = 1.0 if ok else 0.0
                                        else:
                                            uid2any[key] = 1.0 if (ok or uid2any[key] > 0.5) else 0.0

                                    # Update pass@k (shared exp3p instance for both Problem 1 and Problem 2)
                                    allocators: List[ClassDroExp3p] = []
                                    if self.gdro is not None:
                                        allocators.append(self.gdro)
                                    if (
                                        self.rollout_allocator is not None
                                        and self.rollout_allocator is not self.gdro
                                    ):
                                        allocators.append(self.rollout_allocator)
                                    if (
                                        self._offpolicy_classifier_private is not None
                                        and self._offpolicy_classifier_private not in allocators
                                    ):
                                        allocators.append(self._offpolicy_classifier_private)
                                    for allocator in allocators:
                                        if hasattr(allocator, "update_with_passk"):
                                            allocator.update_with_passk(
                                                list(uid2any.keys()), list(uid2any.values())
                                            )
                            except Exception:
                                pass
                            finally:
                                # Clear rollout step state after update
                                if self.rollout_budget_mode == "groupdro":
                                    self._rollout_step_state = {}

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
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

                    # Update curriculum sampler if present
                    train_sampler = getattr(self.train_dataloader, "sampler", None)
                    if hasattr(train_sampler, "update"):
                        train_sampler.update(batch)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        # Print validation metrics at every validation step
                        print(f"Validation metrics at step {self.global_steps}: {val_metrics}")
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
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

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
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
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
