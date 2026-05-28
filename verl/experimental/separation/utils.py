# Copyright 2025 Meituan Ltd. and/or its affiliates
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


import ray

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic resource pool
    training_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]
    if any(role in roles for role in training_roles):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        for role in training_roles:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

    if Role.RewardModel in roles:
        rm_cfg = config.reward.reward_model
        assert rm_cfg.n_gpus_per_node > 0, "config.reward.reward_model.n_gpus_per_node must be greater than 0"
        assert rm_cfg.nnodes > 0, "config.reward.reward_model.nnodes must be greater than 0"

    # FSDP trainer workers must reserve whole GPUs (max_colocate_count=1). The default
    # value of 3 assigns num_gpus=1/3 per worker, so Ray still sees free GPU capacity on
    # the trainer node and may schedule standalone SGLang rollout there → GPU conflict.
    actor_strategy = getattr(getattr(config, "actor_rollout_ref", None), "actor", None)
    strategy = getattr(actor_strategy, "strategy", "") if actor_strategy is not None else ""
    max_colocate_count = 1 if strategy in ("fsdp", "fsdp2") else 3

    return ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
        max_colocate_count=max_colocate_count,
    )


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Always use the unified model engine worker implementation.
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.engine_workers import TrainingWorker

    ray_worker_group_cls = RayWorkerGroup

    train_role = Role.Actor
    if config.get("async_training", {}).get("use_trainer_do_validate", False):
        train_role = Role.ActorRollout

    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Critic: ray.remote(TrainingWorker),
    }

    # Add reference policy (if KL loss or reward is required)
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls
